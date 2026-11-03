export class SurfaceDemandStats {
  constructor() {
    this.levels = new WeakMap();
  }

  sample(presentation) {
    let required = 0,
      missing = 0;
    if (presentation === null) return { required, missing };
    for (const level of presentation.resources.levels.shown) {
      let cached = this.levels.get(level);
      if (
        cached?.addresses !== level.addresses ||
        cached.atlas !== level.atlas ||
        cached.pageTable !== level.pageTable ||
        cached.writes !== level.pageTable.writeCount
      ) {
        let resident = 0;
        for (const { chunk_ZYX } of level.addresses) {
          const entry = level.entry(chunk_ZYX);
          if (entry !== undefined && level.atlas.isResident(entry)) resident++;
        }
        cached = {
          addresses: level.addresses,
          atlas: level.atlas,
          pageTable: level.pageTable,
          writes: level.pageTable.writeCount,
          resident,
        };
        this.levels.set(level, cached);
      }
      required += level.addresses.length;
      missing += level.addresses.length - cached.resident;
    }
    return { required, missing };
  }
}
