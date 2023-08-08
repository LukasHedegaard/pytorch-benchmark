# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), with the exception that v0.X updates include backwards-incompatible API changes.
From v1.0.0 and on, the project will adherence strictly to Semantic Versioning.


## [Unreleased]

## [0.2.1] - 2022-02-11
### Fixed
- Updated on-gpu model benchmaking with best-practices on `cuda.Event` and `cuda.synchronize`.


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
