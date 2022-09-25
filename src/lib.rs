//! Parser and handler for MIME types.

#![no_std]
#![forbid(
    unsafe_code,
    future_incompatible,
    missing_copy_implementations,
    missing_debug_implementations,
    missing_docs
)]

#[cfg(feature = "std")]
extern crate std;

#[rustfmt::ignore]
mod segments;
pub use segments::constants;
use segments::{Subtype, Suffix, Type};

use core::cell::Cell;
use core::cmp;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::iter::FusedIterator;
use core::str::FromStr;
use core::write;

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
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSlash => write!(f, "no slash in MIME type"),
            Self::MissingType => write!(f, "missing MIME type"),
            Self::MissingSubtype => write!(f, "missing MIME subtype"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ParseError {}

/// A MIME type.
#[derive(Clone, Copy)]
pub struct Mime<'a>(Repr<'a>);

impl<'a> fmt::Display for Mime<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.r#type(), self.subtype())?;

        if let Some(suffix) = self.suffix() {
            write!(f, "+{}", suffix)?;
        }

        for (key, value) in self.parameters() {
            write!(f, ";{}={}", key, value)?;
        }

        Ok(())
    }
}

impl<'a> fmt::Debug for Mime<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct Parameters<I>(Cell<Option<I>>);

        impl<'a, 'b, I: Iterator<Item = (&'a str, &'b str)>> fmt::Debug for Parameters<I> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let iter = self.0.take().unwrap();
                f.debug_map().entries(iter).finish()
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
    pub fn new(
        ty: &'a str,
        subtype: &'a str,
        suffix: Option<&'a str>,
        parameters: &'a [(&'a str, &'a str)],
    ) -> Self {
        Self(Repr::Parts {
            ty: Name::new(ty),
            subtype: Name::new(subtype),
            suffix: suffix.map(Name::new),
            parameters,
        })
    }

    /// Create a new MIME type parsed from a string.
    pub fn parse(source: &'a str) -> Result<Self, ParseError> {
        let slash = source.find('/').ok_or(ParseError::NoSlash)?;
        let plus = source.find('+');
        let semicolon = source.find(';');

        if slash == 0 {
            return Err(ParseError::MissingType);
        } else if slash == source.len() - 1 {
            return Err(ParseError::MissingSubtype);
        }

        // Immediately parse it now if there are no parameters.
        if let Some(semicolon) = semicolon {
            // It's difficult to represent parameters without allocation, just store the string.
            Ok(Self(Repr::Buffer {
                buffer: source,
                slash,
                plus,
                semicolon,
            }))
        } else {
            // Intern the parts if possible.
            Ok(Self(Repr::Parts {
                ty: Name::new(&source[..slash]),
                subtype: Name::new(&source[&slash + 1..plus.unwrap_or(source.len())]),
                suffix: plus.map(|plus| Name::new(&source[plus + 1..])),
                parameters: &[],
            }))
        }
    }

    /// Get the type of this MIME type.
    pub fn r#type(&self) -> &str {
        self.type_name().into_str()
    }

    /// Get the subtype of this MIME type.
    pub fn subtype(&self) -> &str {
        self.subtype_name().into_str()
    }

    /// Get the suffix of this MIME type.
    pub fn suffix(&self) -> Option<&str> {
        self.suffix_name().map(|s| s.into_str())
    }

    /// Iterate over the parameters of this MIME type.
    pub fn parameters(&self) -> impl DoubleEndedIterator<Item = (&str, &str)> + FusedIterator {
        match self.0 {
            Repr::Parts { parameters, .. } => Either::Left(parameters.iter().copied()),
            Repr::Buffer {
                buffer, semicolon, ..
            } => Either::Right({
                // Get an iterator over the position of every semicolon in the buffer.
                let semicolons = buffer[semicolon + 1..].split(';');

                semicolons.map(|semicolon| {
                    let mut parts = semicolon.split('=');
                    let key = parts.next().unwrap();
                    let value = parts.next().unwrap();
                    (key, value)
                })
            }),
        }
    }

    /// Get the "essence" of this MIME type.
    ///
    /// The resulting MIME type only contains the type and the subtype, without the suffix or
    /// the parameters.
    pub fn essence(&self) -> Mime<'a> {
        match self.0 {
            Repr::Parts { ty, subtype, .. } => Mime(Repr::Parts {
                ty,
                subtype,
                suffix: None,
                parameters: &[],
            }),
            Repr::Buffer {
                buffer,
                slash,
                plus,
                semicolon,
            } => {
                let end = plus.unwrap_or(semicolon);

                Self::new(&buffer[..slash], &buffer[slash + 1..end], None, &[])
            }
        }
    }

    fn type_name(&self) -> Name<'a, Type> {
        match self.0 {
            Repr::Parts { ty, .. } => ty,
            Repr::Buffer { buffer, slash, .. } => Name::Dynamic(&buffer[..slash]),
        }
    }

    fn subtype_name(&self) -> Name<'a, Subtype> {
        match self.0 {
            Repr::Parts { subtype, .. } => subtype,
            Repr::Buffer {
                buffer,
                slash,
                plus,
                semicolon,
            } => {
                // Figure out where the string will end.
                let end = plus.unwrap_or(semicolon);
                Name::Dynamic(&buffer[slash + 1..end])
            }
        }
    }

    fn suffix_name(&self) -> Option<Name<'a, Suffix>> {
        match self.0 {
            Repr::Parts { suffix, .. } => suffix,
            Repr::Buffer {
                buffer,
                plus,
                semicolon,
                ..
            } => {
                // Figure out where the string will end.
                let end = semicolon;
                plus.map(|plus| Name::Dynamic(&buffer[plus + 1..end]))
            }
        }
    }
}

impl<'a, 'b> PartialEq<&'a str> for Mime<'b> {
    /// Compare a MIME type to a string.
    fn eq(&self, other: &&'a str) -> bool {
        let mut other = *other;

        // See if the type names match.
        let ty = self.r#type();
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
        let subtype = self.subtype();
        let subtype_len = subtype.len();

        if !other.starts_with(subtype) {
            return false;
        }

        // If we have a suffix, the next char is a plus.
        if let Some(suffix) = self.suffix() {
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

            if other != value {
                return false;
            }

            // Advance the string up.
            other = &other[value_len..];
        }

        true
    }
}

impl<'a, 'b> PartialEq<Mime<'a>> for Mime<'b> {
    fn eq(&self, other: &Mime<'a>) -> bool {
        // All of these comparisons are case insensitive at worst and use interned values at best.
        (self.type_name() == other.type_name())
            .and_then(|| self.subtype_name() == other.subtype_name())
            .and_then(|| self.suffix_name() == other.suffix_name())
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
        self.type_name()
            .cmp(&other.type_name())
            .and_then(|| self.subtype_name().cmp(&other.subtype_name()))
            .and_then(|| self.suffix_name().cmp(&other.suffix_name()))
            .and_then(|| cmp_params_ignore_case(self.parameters(), other.parameters()))
    }
}

impl<'a> Hash for Mime<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.type_name().hash(state);
        self.subtype_name().hash(state);
        self.suffix_name().hash(state);
        for (key, value) in self.parameters() {
            hash_ignore_case(key, state);
            value.hash(state);
        }
    }
}

/// Inner representation for the MIME type.
#[derive(Clone, Copy)]
enum Repr<'a> {
    /// We have parts of the string already parsed out.
    ///
    /// This is preferred because it allows us to intern the `&str` slices for
    /// efficiency; however, it cannot accurately represent parameters for parsing
    /// purposes.
    Parts {
        /// The type of the MIME type.
        ty: Name<'a, Type>,

        /// The subtype of the MIME type.
        subtype: Name<'a, Subtype>,

        /// The suffix of the MIME type.
        suffix: Option<Name<'a, Suffix>>,

        /// The parameters of the MIME type.
        parameters: &'a [(&'a str, &'a str)],
    },

    /// We keep a buffer of the original string, and a pointer to the
    /// first slash in the string.
    ///
    /// Since this is only ever used when we need to lazily parse the
    /// parameters, the `semicolon` is not optional. The `parse()` function
    /// eagerly parses MIME types without parameters into `Parts`.
    Buffer {
        /// The original string.
        buffer: &'a str,

        /// The index of the first slash in the string.
        slash: usize,

        /// The index of the first plus in the string.
        plus: Option<usize>,

        /// The index of the first semicolon in the string.
        semicolon: usize,
    },
}

/// Either an interned string or a dynamic string.
#[derive(Clone, Copy)]
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

impl<'a, T: FromStr<Err = InvalidName>> Name<'a, T> {
    fn new(name: &'a str) -> Self {
        match name.parse::<T>() {
            Ok(interned) => Name::Interned(interned),
            Err(_) => Name::Dynamic(name),
        }
    }
}

impl<'a, T: AsRef<str> + PartialEq> PartialEq for Name<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Interned(this), Self::Interned(other)) => this == other,
            (Self::Dynamic(s), Self::Interned(i)) | (Self::Interned(i), Self::Dynamic(s)) => {
                s.eq_ignore_ascii_case(i.as_ref())
            }
            (Self::Dynamic(this), Self::Dynamic(other)) => this.eq_ignore_ascii_case(other),
        }
    }
}

impl<'a, T: AsRef<str> + Eq> Eq for Name<'a, T> {}

// Comparisons between the interned versions are sound because they're sorted in alphabetical order.
impl<'a, T: AsRef<str> + PartialOrd> PartialOrd for Name<'a, T> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        match (self, other) {
            (Self::Interned(this), Self::Interned(other)) => this.partial_cmp(other),
            (Self::Dynamic(s), Self::Interned(i)) | (Self::Interned(i), Self::Dynamic(s)) => {
                Some(cmp_str_ignore_case(s, i.as_ref()))
            }
            (Self::Dynamic(this), Self::Dynamic(other)) => Some(cmp_str_ignore_case(this, other)),
        }
    }
}

impl<'a, T: AsRef<str> + Ord> Ord for Name<'a, T> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        match (self, other) {
            (Self::Interned(this), Self::Interned(other)) => this.cmp(other),
            (Self::Dynamic(s), Self::Interned(i)) | (Self::Interned(i), Self::Dynamic(s)) => {
                cmp_str_ignore_case(s, i.as_ref())
            }
            (Self::Dynamic(this), Self::Dynamic(other)) => cmp_str_ignore_case(this, other),
        }
    }
}

impl<'a, T: AsRef<str> + Hash> Hash for Name<'a, T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_ignore_case(self.as_ref(), state)
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
    left: impl Iterator<Item = (&'a str, &'b str)>,
    right: impl Iterator<Item = (&'c str, &'d str)>,
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
    extern crate alloc;

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
#[derive(Debug, PartialEq, Eq)]
struct InvalidName;

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
