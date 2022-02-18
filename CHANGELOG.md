# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), with the exception that v0.X updates include backwards-incompatible API changes.
From v1.0.0 and on, the project will adherence strictly to Semantic Versioning.


## [Unreleased]


## [0.3.2] - - 2022-02-18

### Fixed
- Warm up batch size


## [0.3.1] - 2022-02-17
### Removed
- `try_custom_warmup`.

### Added
- `warm_up_fn` overload option.
- Support for FLOPs count in torch.nn.Module with input other than Tensor.


## [0.3.0] - 2022-02-15
### Added
- Memory measurement for each batch size.
- Repeated energy measurement.

### Changed
- Number formatting to use u instead of µ.


## [0.2.2] - 2022-02-14
### Added
- Option to redirect info prints.


## [0.2.1] - 2022-02-14
### Fixed
- Added missing with torch.no_grad


## [0.2.0] - 2022-02-11
### Added
- Overloads for benchmark parameters and functions to allow benchmark of custom classes.


## [0.1.2] - 2022-02-10
### Fixed
- GPU compatibility.

### Removed
- Carbon-tracker energy measurement. Library is still too immature at this point.


## [0.1.1] - 2022-02-10
### Added
- Initial version.
