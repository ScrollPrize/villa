# Atlas Implementation

- Background: VC3D atlas is planned as a 2D object canvas over a segment slice view, using atlas coordinates `(winding, y)`.
- Current status:
  - Atlas workspace tab exists beside Main and Lasagna.
  - Center view is an `xy plane` segment slice viewer placeholder/background.
  - Atlas Overview dock exists with an empty atlas list placeholder.
  - Atlas Object Search dock exists as an empty placeholder.
  - Matching Atlas Overview/Search docks are available in the Main workspace via the View menu.
- Not implemented yet:
  - Atlas data model, atlas selection, object projection, links, object search, context-menu seeding, and atlas layout persistence.
- Maintenance:
  - Update this file whenever atlas code or atlas behavior changes.
