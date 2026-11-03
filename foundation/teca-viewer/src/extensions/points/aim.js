import { readTarget, startTarget } from "./targets.js";

const DEFAULT_SPACING = 5;

export class Aim {
  constructor(saved, save, failed) {
    this.save = save;
    this.target = startTarget();
    this.previous = this.target;
    this.only = false;
    this.following = true;
    this.linked = false;
    this.spacings = {
      fiber: DEFAULT_SPACING,
      ink: DEFAULT_SPACING,
      "no ink": DEFAULT_SPACING,
      air: DEFAULT_SPACING,
      tree: DEFAULT_SPACING,
    };
    if (saved === null) return;
    try {
      const target = readTarget(saved.target);
      const previous = readTarget(saved.previous);
      const spacings = { air: DEFAULT_SPACING, ...saved.spacings };
      for (const role of Object.keys(this.spacings))
        if (!(spacings[role] > 0)) {
          const what = role === "tree" ? "a class outside the roles" : role;
          throw new Error(`The address's Points spacing of ${what} is ${JSON.stringify(spacings[role])}.`);
        }
      const flags = { only: saved.only === true, following: saved.following !== false, linked: saved.linked === true };
      Object.assign(this, { target, previous, ...flags });
      Object.assign(this.spacings, spacings);
    } catch (error) {
      failed(error);
    }
  }

  kept() {
    const { target, previous, spacings, only, following, linked } = this;
    this.save({ target, previous, spacings, only, following, linked });
  }

  role() {
    if (this.target.tree !== null) return "tree";
    return this.target.air ? "air" : this.target.role;
  }

  spacing() {
    return this.spacings[this.role()];
  }

  setSpacing(spacing) {
    if (spacing > 0) {
      this.spacings[this.role()] = spacing;
      this.kept();
    }
    return this.spacing();
  }

  showOnly(only) {
    this.only = only;
    this.kept();
  }

  setFollowing(following) {
    this.following = following;
    this.kept();
  }

  setLinked(linked) {
    this.linked = linked;
    this.kept();
  }

  setMark(mark) {
    this.target = { ...this.target, mark };
    this.kept();
  }

  to(next) {
    if (JSON.stringify(next) === JSON.stringify(this.target)) return false;
    this.previous = this.target;
    this.target = next;
    this.kept();
    return true;
  }

  follow(skeleton) {
    if (this.target.tree === null || skeleton.trees.has(this.target.tree)) return;
    this.target = { ...this.target, tree: null };
    this.kept();
  }
}
