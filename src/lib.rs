//! Parser and handler for MIME types.
//!
//! This crate provides a type, [`Mime`], which represents a MIME type as defined in
//! [RFC 2045](https://tools.ietf.org/html/rfc2045) and [RFC 2046](https://tools.ietf.org/html/rfc2046).
//! The aim of this library is to provide strongly typed MIME types that are an overall improvement over
//! just repesenting MIME types as strings.
//!
//! ## Example
//!
//! ```rust
//! use mr_mime::{Mime, constants};
//!
//! // Parse a MIME type from a string.
//! let my_type = Mime::parse("text/html; charset=utf-8").unwrap();
//!
//! // Get the "essence" of a MIME type.
//! let essence = my_type.essence();
//!
//! // Compare it to a wide variety of constants.
//! assert_eq!(essence, constants::TEXT_HTML);
//! ```
//!
//! ## Features
//!
//! This crate has the following features:
//!
//! - `std`, enabled by default, which enables the standard library. This is used to implement
//!   [`std::error::Error`] for [`ParseError`].
//! - `alloc`, enabled by default, which enables the `alloc` crate. This is used to implement
//!   hashing for MIME types. By default, the hashing algorithm tries to use stack space, but for
//!   strings longer than 128 bytes this can lead to a panic. The `alloc` feature ameliorates this
//!   by using the heap instead.

#![no_std]
#![forbid(
    unsafe_code,
    future_incompatible,
    missing_copy_implementations,
    missing_debug_implementations,
    missing_docs
)]
// copied() only stabilized later on
#![allow(clippy::map_clone)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

#[rustfmt::skip]
mod segments;
pub use segments::constants;
use segments::{SubtypeIntern, SuffixIntern, TypeIntern};

use core::cell::Cell;
use core::cmp;
use core::convert::{TryFrom, TryInto};
use core::fmt::{self, Write};
use core::hash::{Hash, Hasher};
use core::iter::FusedIterator;
use core::str::from_utf8;
use core::write;

use memchr::{memchr, memchr2};

macro_rules! matches {
    ($expr: expr, $($pat:pat)|+) => {{
        match ($expr) {
            $($pat)|+ => true,
            _ => false,
        }
    }}
}

/// MIME type parsing error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum ParseError {
    /// There is no slash in the type.
    NoSlash,

    /// The MIME type is missing the type.
    MissingType,

    /// The MIME type is missing the subtype.
    MissingSubtype,

    /// A string contains non-HTTP codepoints.
    NonHttpCodepoints,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::NoSlash => write!(f, "no slash in MIME type"),
            ParseError::MissingType => write!(f, "missing MIME type"),
            ParseError::MissingSubtype => write!(f, "missing MIME subtype"),
            ParseError::NonHttpCodepoints => write!(f, "MIME type contains non-HTTP codepoints"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ParseError {}

/// A MIME type.
///
/// See the [crate-level documentation](../index.html) for more information.
#[derive(Clone, Copy)]
pub struct Mime<'a> {
    /// The MIME type catagory.
    ty: Type<'a>,

    /// The MIME subtype.
    subtype: Subtype<'a>,

    /// The MIME subtype's suffix.
    suffix: Option<Suffix<'a>>,

    /// The MIME parameters.
    parameters: Parameters<'a>,
}

impl<'a> fmt::Display for Mime<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.r#type(), self.subtype())?;

        if let Some(suffix) = self.suffix() {
            write!(f, "+{}", suffix)?;
        }

        for (key, value) in self.parameters() {
            write!(f, ";{}={}", key, FormatQuotedString(value))?;
        }

        Ok(())
    }
}

impl<'a> fmt::Debug for Mime<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct Parameter<'a>(&'a str, &'a [u8]);

        impl<'a> fmt::Debug for Parameter<'a> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct("Parameter")
                    .field("key", &self.0)
                    .field("value", &FormatQuotedString(self.1))
                    .finish()
            }
        }

        struct Parameters<I>(Cell<Option<I>>);

        impl<'a, 'b, I: Iterator<Item = (&'a str, &'b [u8])>> fmt::Debug for Parameters<I> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let iter = self.0.take().unwrap();
                f.debug_list()
                    .entries(iter.map(|(k, v)| Parameter(k, v)))
                    .finish()
            }
        }

        f.debug_struct("Mime")
            .field("type", &self.r#type())
            .field("subtype", &self.subtype())
            .field("suffix", &self.suffix())
            .field(
                "parameters",
                &Parameters(Cell::new(Some(self.parameters()))),
            )
            .finish()
    }
}

impl<'a> Mime<'a> {
    /// Create a new MIME type from its component parts.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use mr_mime::{Mime, constants};
    ///
    /// let my_type = Mime::new(constants::types::TEXT, constants::subtypes::PLAIN, None, &[]);
    /// assert_eq!(my_type, constants::TEXT_PLAIN);
    /// ```
    pub fn new(
        ty: Type<'a>,
        subtype: Subtype<'a>,
        suffix: Option<Suffix<'a>>,
        parameters: &'a [(&'a str, &'a [u8])],
    ) -> Self {
        Self {
            ty,
            subtype,
            suffix,
            parameters: Parameters::Slice(parameters),
        }
    }

    /// Create a new MIME type parsed from a string of bytes.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use mr_mime::{Mime, constants};
    ///
    /// let my_type = Mime::parse_bytes(b"text/plain").unwrap();
    /// assert_eq!(my_type, constants::TEXT_PLAIN);
    /// ```
    pub fn parse_bytes(source: &'a [u8]) -> Result<Self, ParseError> {
        let slash = memchr(b'/', source).ok_or(ParseError::NoSlash)?;
        let plus = memchr(b'+', source);
        let semicolon = memchr(b';', source);

        // Ensure we don't have an empty item.
        let subtype_end = plus.or(semicolon).unwrap_or(source.len());
        if slash == 0 {
            return Err(ParseError::MissingType);
        } else if slash == subtype_end - 1 {
            return Err(ParseError::MissingSubtype);
        }

        // Parse the type.
        let ty = &source[..slash];
        let ty = Type::from_bytes(trim_start(ty)).ok_or(ParseError::NonHttpCodepoints)?;

        // Parse the subtype.
        let subtype = &source[slash + 1..subtype_end];
        let subtype =
            Subtype::from_bytes(trim_end(subtype)).ok_or(ParseError::NonHttpCodepoints)?;

        // Parse the suffix
        let suffix = plus
            .map(|plus| {
                let suffix = &source[plus + 1..semicolon.unwrap_or(source.len())];
                Suffix::from_bytes(trim_end(suffix)).ok_or(ParseError::NonHttpCodepoints)
            })
            .transpose()?;

        // Delay parsing parameters until asked for.
        let parameters = match semicolon {
            None => Parameters::Slice(&[]),
            Some(semicolon) => {
                // Verify that the parameters are valid by parsing them.
                let buffer = &source[semicolon + 1..];
                for (key, value) in parameter_iter(buffer) {
                    // Key should just be HTTP values.
                    let key_valid = key.iter().all(|&b| is_http_codepoint(b));

                    // Value can be HTTP values or quoted strings.
                    let value_valid = if let Some(b'"') = value.first() {
                        value.iter().all(|&b| is_http_quoted_codepoint(b))
                    } else {
                        value.iter().all(|&b| is_http_codepoint(b))
                    };

                    if !key_valid || !value_valid {
                        return Err(ParseError::NonHttpCodepoints);
                    }
                }

                Parameters::Buffer(buffer)
            }
        };

        Ok(Self {
            ty,
            subtype,
            suffix,
            parameters,
        })
    }

    /// Parse this MIME type from a string.
    pub fn parse(source: &'a str) -> Result<Self, ParseError> {
        Self::parse_bytes(source.as_bytes())
    }

    /// Get the type of this MIME type.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use mr_mime::constants;
    ///
    /// assert_eq!(constants::TEXT_PLAIN.r#type(), "text");
    /// ```
    pub fn r#type(&self) -> Type<'_> {
        self.ty
    }

    /// Get the subtype of this MIME type.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use mr_mime::constants;
    ///
    /// assert_eq!(constants::TEXT_PLAIN.subtype(), "plain");
    /// ```
    pub fn subtype(&self) -> Subtype<'_> {
        self.subtype
    }

    /// Get the suffix of this MIME type.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use mr_mime::constants;
    ///
    /// assert_eq!(constants::TEXT_PLAIN.suffix(), None);
    /// assert_eq!(constants::IMAGE_SVG_XML.suffix().map(|s| s.into_str()), Some("xml"));
    /// ```
    pub fn suffix(&self) -> Option<Suffix<'_>> {
        self.suffix
    }

    /// Iterate over the parameters of this MIME type.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use mr_mime::{Mime, constants};
    ///
    /// let mut ty = Mime::parse("text/plain; charset=utf-8").unwrap();
    /// assert_eq!(ty.parameters().count(), 1);
    /// assert_eq!(ty.parameters().next(), Some(("charset", b"utf-8".as_ref())));
    /// ```
    pub fn parameters(&self) -> impl Iterator<Item = (&str, &[u8])> {
        match self.parameters {
            Parameters::Slice(slice) => Either::Left(slice.iter().map(|&(k, v)| (k, v))),
            Parameters::Buffer(buffer) => {
                Either::Right(parameter_iter(buffer).map(|(key, value)| {
                    // Key will always be valid because we parsed it.
                    (from_utf8(key).unwrap(), value)
                }))
            }
        }
    }

    /// Get the "essence" of this MIME type.
    ///
    /// The resulting MIME type only contains the type and the subtype, without the suffix or
    /// the parameters.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use mr_mime::{Mime, constants};
    ///
    /// let my_type = Mime::parse("text/plain;charset=utf-8").unwrap();
    /// assert_eq!(my_type.essence(), constants::TEXT_PLAIN);
    /// ```
    pub fn essence(&self) -> Mime<'a> {
        Mime {
            ty: self.ty,
            subtype: self.subtype,
            suffix: None,
            parameters: Parameters::Slice(&[]),
        }
    }

    /// Calculate the length of this MIME type.
    ///
    /// This returns the length for this given MIME type as if it had been formatted using its
    /// Display trait. This length will thus include any suffix or parameters that it contains. See
    /// essence() to get a slimmed down version of the MIME type.
    pub fn len(&self) -> usize {
        let suffix_length = match self.suffix() {
            Some(s) => s.into_str().len() + 1,
            _ => 0,
        };

        let param_length: usize = self
            .parameters()
            .map(|(k, v)| k.len() + FormatQuotedString(v).len() + 2)
            .sum();

        self.r#type().into_str().len()
            + self.subtype().into_str().len()
            + 1 // slash
            + suffix_length
            + param_length
    }

    /// Checks whether this MIME type is empty or not.
    ///
    /// This function always returns false as it is not possible to construct an empty MIME type.
    /// It is only implemented to make clippy happy.
    pub fn is_empty(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod mime_test {
    use super::*;

    #[test]
    fn mime_len_handles_basic_type() {
        assert_eq!(constants::TEXT_PLAIN.len(), "text/plain".len());
    }

    #[test]
    fn mime_len_handles_suffix() {
        assert_eq!(constants::IMAGE_SVG_XML.len(), "image/svg+xml".len());
    }

    #[test]
    fn mime_len_handles_param() {
        assert_eq!(
            Mime::parse("text/html; charset=utf-8").unwrap().len(),
            "text/html;charset=utf-8".len()
        );
    }

    #[test]
    fn mime_len_handles_multiple_params() {
        assert_eq!(
            Mime::parse("text/html; charset=utf-8; foo=bar")
                .unwrap()
                .len(),
            "text/html;charset=utf-8;foo=bar".len()
        );
    }

    #[test]
    fn mime_len_handles_suffixes_and_params() {
        assert_eq!(
            Mime::parse("image/svg+xml; charset=utf-8; foo=bar")
                .unwrap()
                .len(),
            "image/svg+xml;charset=utf-8;foo=bar".len()
        );
    }
}

impl Mime<'static> {
    /// Guess the MIME type of a file by its extension.
    ///
    /// This library maintains a map of popular extensions to the MIME types that they may
    /// represent. This function preforms a lookup into that list and returns an iterator
    /// over the possible MIME types that the extension may represent.
    ///
    /// Remember that this function only inspects the extension, not the actual contents
    /// of the file. Despite what a file's extension says, it may or may not be a valid
    /// file of that type. For untrusted user input, you should always check the file's
    /// contents to ensure that it is valid.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use mr_mime::{Mime, constants};
    ///
    /// assert_eq!(Mime::guess("html").next(), Some(constants::TEXT_HTML));
    /// ```
    pub fn guess(extension: &str) -> impl ExactSizeIterator<Item = Mime<'static>> + FusedIterator {
        segments::guess_mime_type(extension)
            .unwrap_or(&[])
            .iter()
            .map(|&c| c)
    }
}

impl PartialEq<str> for Mime<'_> {
    /// Compare a MIME type to a string.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use mr_mime::{Mime, constants};
    ///
    /// assert_eq!(constants::TEXT_PLAIN, "text/plain");
    /// ```
    fn eq(&self, mut other: &str) -> bool {
        // See if the type names match.
        let ty = self.r#type().into_str();
        let ty_len = ty.len();

        if !other.starts_with(ty) {
            return false;
        }

        // Next char should be a slash.
        if other.as_bytes()[ty_len] != b'/' {
            return false;
        }

        // Next string should be the subtype.
        other = &other[ty_len + 1..];
        let subtype = self.subtype().into_str();
        let subtype_len = subtype.len();

        if !other.starts_with(subtype) {
            return false;
        }

        // If we have a suffix, the next char is a plus.
        if let Some(suffix) = self.suffix() {
            let suffix = suffix.into_str();
            if other.as_bytes()[subtype_len] != b'+' {
                return false;
            }

            // Next string should be the suffix.
            other = &other[subtype_len + 1..];
            let suffix_len = suffix.len();

            if !other.starts_with(suffix) {
                return false;
            }

            other = &other[suffix_len..];
        } else {
            other = &other[subtype_len..];
        }

        // Now, compare for parameters.
        for (key, value) in self.parameters() {
            // The next char should be a semicolon.
            if other.as_bytes()[0] != b';' {
                return false;
            }

            // Next string should be the key.
            other = &other[1..];
            let key_len = key.len();

            if !other.eq_ignore_ascii_case(key) {
                return false;
            }

            // Next char should be an equals sign.
            if other.as_bytes()[key_len] != b'=' {
                return false;
            }

            // Next string should be the value.
            other = &other[key_len + 1..];
            let value_len = value.len();

            if &other.as_bytes()[..value_len] != value {
                return false;
            }

            // Advance the string up.
            other = &other[value_len..];
        }

        true
    }
}

impl<'a, 'b> PartialEq<&'a str> for Mime<'b> {
    fn eq(&self, other: &&'a str) -> bool {
        self.eq(*other)
    }
}

impl<'a, 'b> PartialEq<Mime<'a>> for Mime<'b> {
    fn eq(&self, other: &Mime<'a>) -> bool {
        // All of these comparisons are case insensitive at worst and use interned values at best.
        (self.r#type() == other.r#type())
            .and_then(|| self.subtype() == other.subtype())
            .and_then(|| self.suffix() == other.suffix())
            .and_then(|| {
                cmp_params_ignore_case(self.parameters(), other.parameters())
                    == cmp::Ordering::Equal
            })
    }
}

impl<'a> Eq for Mime<'a> {}

impl<'a, 'b> PartialOrd<Mime<'a>> for Mime<'b> {
    fn partial_cmp(&self, other: &Mime<'a>) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> Ord for Mime<'a> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.r#type()
            .cmp(&other.r#type())
            .and_then(|| self.subtype().cmp(&other.subtype()))
            .and_then(|| self.suffix().cmp(&other.suffix()))
            .and_then(|| cmp_params_ignore_case(self.parameters(), other.parameters()))
    }
}

impl<'a> Hash for Mime<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.r#type().hash(state);
        self.subtype().hash(state);
        self.suffix().hash(state);
        for (key, value) in self.parameters() {
            hash_ignore_case(key, state);
            value.hash(state);
        }
    }
}

/// Wrapper types for `Name<'a, T>`.
macro_rules! name_wrappers {
    (
        $(
            $(#[$outer:meta])*
            $name: ident <'a> => Name<'a, $ty: ty>
        ),* $(,)?
    ) => {
        $(
            $(#[$outer])*
            #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
            pub struct $name <'a> ( Name<'a, $ty> );

            impl<'a> $name<'a> {
                /// Create a new name from a string.
                pub fn new(s: &'a str) -> Option<Self> {
                    Name::new(s).map($name)
                }

                /// Create a new name from a string of bytes.
                pub fn from_bytes(s: &'a [u8]) -> Option<Self> {
                    Name::from_bytes(s).map($name)
                }

                /// Get the string representation of this name.
                pub fn into_str(self) -> &'a str {
                    self.0.into_str()
                }
            }

            impl fmt::Debug for $name <'_> {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.debug_tuple(stringify!($name))
                        .field(&self.0.into_str())
                        .finish()
                }
            }

            impl fmt::Display for $name <'_> {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.write_str(self.0.into_str())
                }
            }

            impl AsRef<str> for $name <'_> {
                fn as_ref(&self) -> &str {
                    self.0.into_str()
                }
            }

            impl<'a> From<Name<'a, $ty>> for $name<'a> {
                fn from(name: Name<'a, $ty>) -> Self {
                    $name(name)
                }
            }

            impl PartialEq<&str> for $name<'_> {
                fn eq(&self, other: &&str) -> bool {
                    self.0.into_str().eq_ignore_ascii_case(other)
                }
            }

            impl<'a> TryFrom<&'a str> for $name<'a> {
                type Error = ParseError;

                fn try_from(s: &'a str) -> Result<Self, Self::Error> {
                    Self::new(s).ok_or(ParseError::NonHttpCodepoints)
                }
            }

            impl<'a> TryFrom<&'a [u8]> for $name<'a> {
                type Error = ParseError;

                fn try_from(s: &'a [u8]) -> Result<Self, Self::Error> {
                    Self::from_bytes(s).ok_or(ParseError::NonHttpCodepoints)
                }
            }
        )*
    }
}

name_wrappers! {
    /// The type name of a MIME type.
    Type<'a> => Name<'a, TypeIntern>,
    /// The subtype name of a MIME type.
    Subtype<'a> => Name<'a, SubtypeIntern>,
    /// The suffix name of a MIME type.
    Suffix<'a> => Name<'a, SuffixIntern>
}

/// Inner representation for the MIME parameters.
#[derive(Clone, Copy)]
enum Parameters<'a> {
    /// Parameters are given by a slice.
    Slice(&'a [(&'a str, &'a [u8])]),

    /// Parameters are given by a buffer we need to parse on demand.
    Buffer(&'a [u8]),
}

/// Either an interned string or a dynamic string.
#[derive(Debug, Clone, Copy)]
enum Name<'a, Intern> {
    /// An interned string.
    Interned(Intern),

    /// A dynamic string.
    Dynamic(&'a str),
}

impl<'a, T: Into<&'static str>> Name<'a, T> {
    fn into_str(self) -> &'a str {
        match self {
            Name::Interned(interned) => interned.into(),
            Name::Dynamic(dynamic) => dynamic,
        }
    }
}

impl<'a, T> From<T> for Name<'a, T> {
    fn from(item: T) -> Self {
        Name::Interned(item)
    }
}

impl<'a, T: AsRef<str>> AsRef<str> for Name<'a, T> {
    fn as_ref(&self) -> &str {
        match self {
            Name::Interned(interned) => interned.as_ref(),
            Name::Dynamic(dynamic) => dynamic,
        }
    }
}

impl<'a, T: TryFrom<&'a [u8]>> Name<'a, T> {
    fn from_bytes(name: &'a [u8]) -> Option<Self> {
        match name.try_into() {
            Ok(interned) => Some(Name::Interned(interned)),
            Err(_) => {
                // Ensure all bytes are valid HTTP codepoints.
                if !name.iter().all(|&c| is_http_codepoint(c)) {
                    return None;
                }

                // unwrap() here is OK because all HTTP codepoints implies ASCII.
                Some(Name::Dynamic(from_utf8(name).unwrap()))
            }
        }
    }

    fn new(name: &'a str) -> Option<Self> {
        Self::from_bytes(name.as_bytes())
    }
}

impl<'a, T: AsRef<str> + PartialEq> PartialEq for Name<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Name::Interned(this), Name::Interned(other)) => this == other,
            (Name::Dynamic(s), Name::Interned(i)) | (Name::Interned(i), Name::Dynamic(s)) => {
                s.eq_ignore_ascii_case(i.as_ref())
            }
            (Name::Dynamic(this), Name::Dynamic(other)) => this.eq_ignore_ascii_case(other),
        }
    }
}

impl<'a, T: AsRef<str> + Eq> Eq for Name<'a, T> {}

// Comparisons between the interned versions are sound because they're sorted in alphabetical order.
impl<'a, T: AsRef<str> + PartialOrd> PartialOrd for Name<'a, T> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        match (self, other) {
            (Name::Interned(this), Name::Interned(other)) => this.partial_cmp(other),
            (Name::Dynamic(s), Name::Interned(i)) | (Name::Interned(i), Name::Dynamic(s)) => {
                Some(cmp_str_ignore_case(s, i.as_ref()))
            }
            (Name::Dynamic(this), Name::Dynamic(other)) => Some(cmp_str_ignore_case(this, other)),
        }
    }
}

impl<'a, T: AsRef<str> + Ord> Ord for Name<'a, T> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        match (self, other) {
            (Name::Interned(this), Name::Interned(other)) => this.cmp(other),
            (Name::Dynamic(s), Name::Interned(i)) | (Name::Interned(i), Name::Dynamic(s)) => {
                cmp_str_ignore_case(s, i.as_ref())
            }
            (Name::Dynamic(this), Name::Dynamic(other)) => cmp_str_ignore_case(this, other),
        }
    }
}

impl<'a, T: AsRef<str> + Hash> Hash for Name<'a, T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_ignore_case(self.as_ref(), state)
    }
}

/// Get an iterator over the parameters of a MIME type.
///
/// Takes the semicolon-separated list of parameters as a slice of bytes.
fn parameter_iter(bytes: &[u8]) -> impl Iterator<Item = (&[u8], &[u8])> {
    ParameterIter { bytes }
}

struct ParameterIter<'a> {
    /// The bytes to parse.
    bytes: &'a [u8],
}

impl<'a> Iterator for ParameterIter<'a> {
    type Item = (&'a [u8], &'a [u8]);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.bytes.is_empty() {
                return None;
            }

            // Read to the next semicolon or equals sign.
            let posn = memchr2(b';', b'=', self.bytes).unwrap_or(self.bytes.len());

            // Read the parameter name.
            let name = trim_start(&self.bytes[..posn]);
            let symbol = self.bytes.get(posn);
            self.bytes = self.bytes.get(posn + 1..).unwrap_or(&[]);

            if let Some(b';') | None = symbol {
                // No equals sign, so this is a flag parameter.

                if name.is_empty() {
                    // Empty parameter name, so skip it.
                    continue;
                }

                return Some((name, &[]));
            }

            // Is this a quoted string?
            if let Some(b'"') = self.bytes.first() {
                let mut start = 1;
                loop {
                    // Read to the next quote or the next slash.
                    let posn =
                        memchr2(b'"', b'\\', &self.bytes[start..]).unwrap_or(self.bytes.len());

                    // Read the parameter value.
                    match self.bytes.get(posn) {
                        Some(b'"') | None => {
                            // We've reached the end of the quoted string.
                            let value = &self.bytes[1..posn];
                            self.bytes = self.bytes.get(posn + 1..).unwrap_or(&[]);
                            return Some((name, value));
                        }
                        Some(b'\\') => {
                            // We've reached a backslash, so skip the next character.
                            start = posn + 2;
                        }
                        _ => unreachable!(),
                    }
                }
            } else {
                // This isn't a quoted string, just read to the next semicolon.
                let posn = memchr(b';', self.bytes).unwrap_or(self.bytes.len());
                let value = &self.bytes[..posn];
                self.bytes = self.bytes.get(posn + 1..).unwrap_or(&[]);

                return Some((name, value));
            }
        }
    }
}

/// Order two strings, ignoring case.
fn cmp_str_ignore_case(a: &str, b: &str) -> cmp::Ordering {
    let common_len = cmp::min(a.len(), b.len());

    // Get the common part of each string.
    let a_part = &a[..common_len];
    let b_part = &b[..common_len];

    // Compare the common part.
    for (ac, bc) in a_part.chars().zip(b_part.chars()) {
        let ac = ac.to_ascii_lowercase();
        let bc = bc.to_ascii_lowercase();

        match ac.cmp(&bc) {
            cmp::Ordering::Equal => continue,
            other => return other,
        }
    }

    // If the common part is equal, compare the lengths.
    a.len().cmp(&b.len())
}

/// Compare two sets of parameters, ignoring case.
fn cmp_params_ignore_case<'a, 'b, 'c, 'd>(
    left: impl Iterator<Item = (&'a str, &'b [u8])>,
    right: impl Iterator<Item = (&'c str, &'d [u8])>,
) -> cmp::Ordering {
    let mut left = left.fuse();
    let mut right = right.fuse();

    for (left, right) in left.by_ref().zip(right.by_ref()) {
        match cmp_str_ignore_case(left.0, right.0) {
            cmp::Ordering::Equal => {}
            other => return other,
        }

        match left.1.cmp(right.1) {
            cmp::Ordering::Equal => {}
            other => return other,
        }
    }

    if left.next().is_some() {
        cmp::Ordering::Greater
    } else if right.next().is_some() {
        cmp::Ordering::Less
    } else {
        cmp::Ordering::Equal
    }
}

/// Hash a string in such a way that it ignores case.
fn hash_ignore_case(a: &str, state: &mut impl Hasher) {
    #[cfg(feature = "alloc")]
    use alloc::string::String;

    // For our purposes, strings should never be longer than this.
    const MAX_LEN: usize = 128;

    // Convert the string to lowercase on the stack or the heap.
    let mut stack_space = [0u8; MAX_LEN];
    #[cfg(feature = "alloc")]
    let mut heap_space;

    let copied_str = if a.len() > MAX_LEN {
        #[cfg(not(feature = "alloc"))]
        panic!("MIME type string cannot be hashed longer than 128 characters");

        #[cfg(feature = "alloc")]
        {
            heap_space = String::from(a);
            &mut heap_space
        }
    } else {
        stack_space[..a.len()].copy_from_slice(a.as_bytes());
        core::str::from_utf8_mut(&mut stack_space[..a.len()]).unwrap()
    };

    copied_str.make_ascii_lowercase();

    // Hash the lowercase string.
    copied_str.hash(state);
}

/// Is this byte a valid HTTP codepoint?
fn is_http_codepoint(b: u8) -> bool {
    matches!(
        b,
        b'!'
        | b'#'
        | b'$'
        | b'%'
        | b'&'
        | b'\''
        | b'*'
        | b'+'
        | b'-'
        | b'.'
        | b'^'
        | b'_'
        | b'`'
        | b'|'
        | b'~'
        | b'a'..=b'z'
        | b'A'..=b'Z'
        | b'0'..=b'9'
    )
}

/// Is this byte HTTP whitespace?
fn is_http_whitespace(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\r' | b'\n')
}

/// Is this byte valid in an HTTP quoted string?
fn is_http_quoted_codepoint(b: u8) -> bool {
    matches!(b, b'\t' | b' '..=b'~' | 0x80..=0xFF)
}

/// Trim the start of a byte stream of whitespace.
fn trim_start(mut s: &[u8]) -> &[u8] {
    while let Some((b, rest)) = s.split_first() {
        if !is_http_whitespace(*b) {
            break;
        }

        s = rest;
    }

    s
}

/// Trim the end of a byte stream of whitespace.
fn trim_end(mut s: &[u8]) -> &[u8] {
    while let Some((b, rest)) = s.split_last() {
        if !is_http_whitespace(*b) {
            break;
        }

        s = rest;
    }

    s
}

/// Monad for making comparisons slightly easier.
trait Comparison: Sized {
    /// Take these two comparisons together.
    fn and_then(self, other: impl FnOnce() -> Self) -> Self;
}

impl Comparison for bool {
    fn and_then(self, other: impl FnOnce() -> Self) -> Self {
        match self {
            true => other(),
            false => false,
        }
    }
}

impl Comparison for Option<cmp::Ordering> {
    fn and_then(self, other: impl FnOnce() -> Self) -> Self {
        match self {
            Some(cmp::Ordering::Greater) => other(),
            this => this,
        }
    }
}

impl Comparison for cmp::Ordering {
    fn and_then(self, other: impl FnOnce() -> Self) -> Self {
        if let cmp::Ordering::Equal = self {
            other()
        } else {
            self
        }
    }
}

/// Error for generated code to use for unmatched names.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct InvalidName;

#[derive(Debug)]
enum Either<A, B> {
    Left(A),
    Right(B),
}

impl<A, B> Iterator for Either<A, B>
where
    A: Iterator,
    B: Iterator<Item = A::Item>,
{
    type Item = A::Item;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Either::Left(a) => a.next(),
            Either::Right(b) => b.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Either::Left(a) => a.size_hint(),
            Either::Right(b) => b.size_hint(),
        }
    }

    fn fold<Closure, F>(self, init: Closure, f: F) -> Closure
    where
        Self: Sized,
        F: FnMut(Closure, Self::Item) -> Closure,
    {
        match self {
            Either::Left(a) => a.fold(init, f),
            Either::Right(b) => b.fold(init, f),
        }
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        match self {
            Either::Left(a) => a.nth(n),
            Either::Right(b) => b.nth(n),
        }
    }

    fn last(self) -> Option<Self::Item>
    where
        Self: Sized,
    {
        match self {
            Either::Left(a) => a.last(),
            Either::Right(b) => b.last(),
        }
    }
}

impl<A, B> FusedIterator for Either<A, B>
where
    A: FusedIterator,
    B: FusedIterator<Item = A::Item>,
{
}

impl<A, B> ExactSizeIterator for Either<A, B>
where
    A: ExactSizeIterator,
    B: ExactSizeIterator<Item = A::Item>,
{
}

impl<A, B> DoubleEndedIterator for Either<A, B>
where
    A: DoubleEndedIterator,
    B: DoubleEndedIterator<Item = A::Item>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        match self {
            Either::Left(a) => a.next_back(),
            Either::Right(b) => b.next_back(),
        }
    }

    fn rfold<Closure, F>(self, init: Closure, f: F) -> Closure
    where
        Self: Sized,
        F: FnMut(Closure, Self::Item) -> Closure,
    {
        match self {
            Either::Left(a) => a.rfold(init, f),
            Either::Right(b) => b.rfold(init, f),
        }
    }

    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        match self {
            Either::Left(a) => a.nth_back(n),
            Either::Right(b) => b.nth_back(n),
        }
    }
}

/// Invariant: `0` is either:
///
/// - An ASCII string.
/// - An HTTP quoted string.
struct FormatQuotedString<'a>(&'a [u8]);

impl<'a> FormatQuotedString<'a> {
    pub fn len(&self) -> usize {
        self.0
            .iter()
            .flat_map(|&c| core::ascii::escape_default(c))
            .count()
    }
}

impl<'a> fmt::Display for FormatQuotedString<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for ch in self.0.iter().flat_map(|&c| core::ascii::escape_default(c)) {
            f.write_char(ch as char)?;
        }

        Ok(())
    }
}

impl<'a> fmt::Debug for FormatQuotedString<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[cfg(test)]
mod fqs_test {
    use super::*;

    #[test]
    fn fqs_len_handles_empty_array() {
        assert_eq!(FormatQuotedString(&[]).len(), 0);
    }

    #[test]
    fn fqs_len_handles_quoted_string() {
        let input =
            b"this%20is%20http%20encoded%E2%80%A6%20or%20is%20it%3F%20%C5%B6%C4%99%C5%A1%20it%20is";
        assert_eq!(FormatQuotedString(input).len(), 84);
    }

    #[test]
    fn fqs_len_handles_standard_ascii() {
        let input = b"this is not encoded or special at all";
        assert_eq!(FormatQuotedString(input).len(), 37);
    }

    #[test]
    fn fqs_len_handles_utf8() {
        let input = b"\xC5\xB6'\"\\";
        assert_eq!(FormatQuotedString(input).len(), 14);
    }
}
