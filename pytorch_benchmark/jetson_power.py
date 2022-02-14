#!/usr/bin/python3
# Modified version of https://github.com/opendr-eu/jetson_power

import os
import subprocess
import sys
import threading
import time
from logging import getLogger
from time import sleep

logger = getLogger("pytorch_benchmark")


class PowerEstimator:
    def __init__(
        self,
        idle_load_duration=3,
        idle_load_samples=10,
        sampling_rate=30,
        print_fn=logger.debug,
    ):
        """
        Creates a PowerEstimator object for measuring energy consumption
        :param idle_load_duration: Total time duration for estimating idle energy consumption
        :param idle_load_samples: Number of samples to be used for idle energy consumption estimation
        :param sampling_rate: Sampling rate (Hz) to be used when estimating the energy consumption
        """
        self.print_fn = print_fn

        self.devices = []
        self._device_scan("/sys/bus/i2c/drivers/ina3221x/")
        # self._device_scan('/sys/devices/3160000.i2c/i2c-0/')

        assert len(self.devices) > 0, "Device scan did not find any Jetson devices"
        self.print_fn("Jetson PowerLogger found %d power devices" % (len(self.devices)))

        self.idle_load = 0
        self._estimate_standby_load(idle_load_duration, idle_load_samples)
        self.sampling_interval = 1.0 / sampling_rate

    def _device_scan(self, base_path):
        files = ["iio_device/in_power" + str(i) + "_input" for i in range(5)]
        for _, folders, _ in os.walk(base_path):
            for folder in folders:
                for file in files:
                    cur_path = os.path.join(base_path, folder, file)
                    if os.path.exists(cur_path):
                        self.devices.append(cur_path)
                        self.print_fn("Device found @", cur_path)
                    elif os.path.exists(cur_path.replace("iio_device", "iio:device0")):
                        self.devices.append(
                            cur_path.replace("iio_device", "iio:device0")
                        )
                        self.print_fn(
                            "Device found @",
                            cur_path.replace("iio_device", "iio:device0"),
                        )
                    for i in range(1, 5):
                        if os.path.exists(
                            cur_path.replace("iio_device", "iio:device" + str(i))
                        ):
                            self.devices.append(
                                cur_path.replace("iio_device", "iio:device" + str(i))
                            )
                            logger.debug(
                                "Device found @",
                                cur_path.replace("iio_device", "iio:device" + str(i)),
                            )

    def _get_instant_power(self):
        total_power = 0
        for device in self.devices:
            with open(device, "r") as f:
                total_power += float(f.readline())
        return total_power

    def _estimate_standby_load(self, duration, samples=10.0):
        logger.info(
            "Jetson PowerLogger estimating background power load for %d s, please do not use the system."
            % (duration)
        )
        idle_load = 0.0
        for i in range(samples):
            idle_load += self._get_instant_power()
            sleep(duration / float(samples))
        self.idle_load = idle_load / samples
        logger.info("Jetson PowerLogger idle power: %7.3f mW" % self.idle_load)

    def estimate_cmd_power(self, cmd):
        start = time.time()
        total_energy = 0
        total_energy_over_idle = 0
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        while True:
            sleep(self.sampling_interval)
            poll_start_time = time.time()
            poll = p.poll()
            if poll is None:
                current_power = self._get_instant_power()
                poll_time = time.time() - poll_start_time
                total_energy += current_power * (self.sampling_interval + poll_time)
                total_energy_over_idle += (current_power - self.idle_load) * (
                    self.sampling_interval + poll_time
                )
            else:
                break
        total_time = time.time() - start
        return total_energy, total_energy_over_idle, total_time

    def estimate_fn_power(self, fn):
        start = time.time()
        total_energy = 0
        total_energy_over_idle = 0
        th = threading.Thread(target=fn)
        th.start()

        while True:
            sleep(self.sampling_interval)
            poll_start_time = time.time()
            if th.is_alive():
                current_power = self._get_instant_power()
                poll_time = time.time() - poll_start_time
                total_energy += current_power * (self.sampling_interval + poll_time)
                total_energy_over_idle += (current_power - self.idle_load) * (
                    self.sampling_interval + poll_time
                )
            else:
                break
        total_time = time.time() - start
        return total_energy, total_energy_over_idle, total_time


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error(
            "Please provide a program to run as argument, e.g., ./p_est stress --cpu 6 -t 5"
        )
        sys.exit(1)

    p_est = PowerEstimator()
    total_energy, total_energy_over_idle, total_time = p_est.estimate_cmd_power(
        sys.argv[1:]
    )

    logger.info("Total energy consumption was = %7.3f J" % (total_energy / 1000.0))
    logger.info(
        "Total energy over baseline was = %7.3f J" % (total_energy_over_idle / 1000.0)
    )
    logger.info("Total time running was = %5.2f s" % total_time)
