# Future half-resolution rendering

Change renderer defaults and full-render preset configs from4 to8baseUV/pixel
and from8 to4pixel fiber width. Keep same UV bounds,CT source level and output
formats. Existing outputs/frozen run configs stay unchanged; overlay-only refresh
on an old image keeps its explicitly requested width. No solve/rerender needed.

## Spec/docs/tests
Record future resolution policy in spec/docs/changelog. Test half-size canvas,
same coordinate mapping at shared pixel locations,and proportional line width.
Review independently, run renderer tests, archive preceding active task log.
