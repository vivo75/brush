//! Post-expansion assignment detection for declaration builtins.
//!
//! The parser only recognises an assignment word when the name is a
//! **literal** identifier (`brush-parser/src/word.rs`, `assigned_scalar_name`),
//! so `export ${var}=value` arrives at the builtin as a
//! [`brush_core::CommandArg::String`]. bash performs the assignment
//! **after** expansion for declaration builtins (`export` / `declare` /
//! `local` / `readonly` / `typeset`; POSIX XCU 2.9.1), so the builtins
//! re-examine an expanded string operand here: split on the first `=`
//! and treat the left side as the name.
//!
//! This is the small, builtin-local half of the choice the thin-fork fix
//! records: upstream [#1280](https://github.com/reubeno/brush/pull/1280)
//! reworks assignment expansion wholesale (word + subscript passes across
//! `interp.rs`/`expansion.rs`, a `declare.rs` split, breaking API changes)
//! and is still a Draft. Re-examining the already-expanded string inside
//! the builtins fixes the silent no-op without touching the expansion
//! pipeline, so it rebases onto #1280 as a deletion rather than a conflict.

/// An expanded declaration operand that turned out to be an assignment.
pub(crate) struct ExpandedAssignment {
    /// Text before the first `=` (without a trailing `+`, if any).
    pub(crate) name: String,
    /// Whether the operand used the `+=` append form.
    pub(crate) append: bool,
    /// Text after the first `=`, already expanded; used verbatim.
    pub(crate) value: String,
}

impl ExpandedAssignment {
    /// Re-wrap as the assignment the parser would have produced for a
    /// literal name, so the builtin's existing `CommandArg::Assignment`
    /// path handles it. The value is already expanded and must not be
    /// expanded again: a literal [`brush_parser::ast::Word`] flattens back
    /// to itself.
    #[cfg(feature = "builtin.export")]
    pub(crate) fn into_command_arg(self) -> brush_core::CommandArg {
        brush_core::CommandArg::Assignment(brush_core::parser::ast::Assignment {
            name: brush_core::parser::ast::AssignmentName::VariableName(self.name),
            value: brush_core::parser::ast::AssignmentValue::Scalar(self.value.into()),
            append: self.append,
            loc: brush_core::SourceSpan::default(),
        })
    }
}

/// Split an expanded declaration operand on the first `=`.
///
/// Returns `None` when there is no `=` at all — the operand is a plain
/// name (`export ${var}` still just marks exported) — and when the value
/// is a compound array literal (`name=(…)`): only scalar assignments are
/// recognised here, so e.g. `declare +a 'arr=(3 4)'` keeps the builtin's
/// existing handling (bash refuses the conversion; it never becomes a
/// scalar assignment of the literal text). The caller validates the name:
/// an invalid one is bash's diagnostic, not a silent no-op.
pub(crate) fn split_expanded_assignment(s: &str) -> Option<ExpandedAssignment> {
    let (left, value) = s.split_once('=')?;
    if value.starts_with('(') {
        return None;
    }
    let (name, append) = left
        .strip_suffix('+')
        .map_or((left, false), |name| (name, true));
    Some(ExpandedAssignment {
        name: name.to_owned(),
        append,
        value: value.to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_the_first_equals() {
        let assignment = split_expanded_assignment("CC=gcc -E -P").unwrap();
        assert_eq!(assignment.name, "CC");
        assert!(!assignment.append);
        assert_eq!(assignment.value, "gcc -E -P");

        let assignment = split_expanded_assignment("EQ=a=b").unwrap();
        assert_eq!(assignment.name, "EQ");
        assert_eq!(assignment.value, "a=b");
    }

    #[test]
    fn detects_the_append_form() {
        let assignment = split_expanded_assignment("MYVAR+=more").unwrap();
        assert_eq!(assignment.name, "MYVAR");
        assert!(assignment.append);
        assert_eq!(assignment.value, "more");
    }

    #[test]
    fn no_equals_is_not_an_assignment() {
        assert!(split_expanded_assignment("CC").is_none());
    }

    #[test]
    fn compound_array_value_is_not_a_scalar_assignment() {
        assert!(split_expanded_assignment("arr=(3 4)").is_none());
        assert!(split_expanded_assignment("arr+=(3 4)").is_none());
    }

    #[test]
    fn empty_name_still_splits_for_the_caller_to_reject() {
        let assignment = split_expanded_assignment("=v").unwrap();
        assert_eq!(assignment.name, "");
        assert_eq!(assignment.value, "v");
    }
}
