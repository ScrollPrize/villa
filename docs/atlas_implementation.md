# Atlas Implementation

- Background: VC3D atlas is planned as a 2D object canvas over a segment slice view, using atlas coordinates `(winding, y)`.
- Current status:
  - Atlas workspace tab exists beside Main and Lasagna.
  - Atlas display loads saved atlas base meshes and mapped fiber overlays.
  - Atlas Overview lists saved atlases and mapped fiber counts.
  - Atlas Object Search can search mapped fibers against saved fibers in original volume coordinates.
  - Matching Atlas Overview/Search docks are available in the Main workspace via the View menu.
- Not implemented yet:
  - Persisted intersection links, context-menu seeding, and atlas layout persistence.
- Maintenance:
  - Update this file whenever atlas code or atlas behavior changes.
