"""Selected native objects with their record, geometry and complete identity together."""
from collections.abc import Mapping
from ._native import object_reference


class SelectedRecord(Mapping):
    def __init__(self, record):
        self.record = record

    def __getitem__(self, key):
        return self.record[key]

    def __iter__(self):
        return iter(self.record)

    def __len__(self):
        return len(self.record)

    @property
    def reference(self):
        return object_reference(self.record)

    @property
    def uuid(self):
        return self.reference['object']

    @property
    def version(self):
        return self.reference['version']

    @property
    def kind(self):
        return self.record['kind']

    @property
    def name(self):
        return self.record['name']

    @property
    def meta(self):
        return self.record['meta']

    @property
    def tags(self):
        return self.record.get('tags', [])

    @property
    def fiber(self):
        return self.record.get('fiber')

    @property
    def track(self):
        return self.record.get('track')


class SelectedObject(SelectedRecord):
    def __init__(self, revision):
        super().__init__(revision['record'])
        self.revision = revision

    @property
    def geometry_revision(self):
        return self.revision['geometry_revision']
