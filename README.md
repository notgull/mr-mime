# `mr-mime`

`mr-mime` is a library for parsing and generating MIME messages. It is created due to my dissatisfaction with the current MIME library used in most Rust projects, [`mime`]. While `mr-mime` is not a drop-in replacement, the API is very similar and overall aims to reduce some notable warts found in [`mime`].

[`mime`]: https://crates.io/crates/mime

## Improvements over `mime`

* `mr-mime` is not only `no_std`, but can operate without an allocator. This means that it can be used in `#![no_std]` environments, and in environments where the allocator is not available (e.g. embedded systems).
* `mr-mime` is `forbid(unsafe_code)`, meaning that is contains no unsafe code. This reduces the potential surface where a memory vulnerability can occur.
* `mr-mime` interns and provides constants for a wider variety of MIME types.

## License

`mr-mime` is licensed under one of the following licenses, at your option:

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
