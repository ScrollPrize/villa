# Plan

1. Add independent raw-volume and segment filters, preserving existing selection semantics.
2. Add fresh project name/destination options to shared creation: bypass cached full-sample projects.
3. Create Project beside Open Sample: initially unchecked resources grouped by
   representation type; checkable individual entries and select all/none, name and destination browser;
   confirm overwrite, then use existing asynchronous opening and session-save gates.
4. Test empty selection, independent predictions, segment filtering, saved name/path
   and isolation from cached projects. Verify immediate opening, cancellation and
   save failure without closing the active project. Build VC3D and run catalog tests.
5. Preserve representation coordinate metadata and channel attachments, but do
   not attach unselected scans or rebased source views in selective creation.
   Selected segments attach individual directories, not aggregate cache
   roots, and must not mark unselected cached segments orphaned.

## Spec update

Follow-up: extract annotation fiber-path naming into one shared helper, use it
in catalog and ordinary new-project creation, and test empty-directory creation
and preservation of existing contents.

Document selective creation, empty defaults, required coordinate dependencies and global cache reuse.

## Docs updates

Add catalog usage documentation and validation log.

## Changelog

Record new selective project creation action.

## Review

Independent review against task, spec and overarching plan before implementation.
Reviewer identified dependency semantics and cancellation/save-failure coverage;
these are explicit above. The render efficiency plan remains unchanged.
