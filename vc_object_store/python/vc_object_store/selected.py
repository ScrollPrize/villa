"""Selected native objects with their record, geometry and complete identity together."""
from ._native import object_reference
from dataclasses import dataclass


class SelectedRecord:
    def __init__(self, record):
        self.record = record

    @property
    def reference(self):
        return object_reference(self.record)

    @property
    def uuid(self):
        return self.record['uuid']

    @property
    def version(self):
        return self.record['version']

    @property
    def retired(self):
        return self.record.get('retired', False)

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


@dataclass
class SelectedObject(SelectedRecord):
    revision: dict

    @property
    def record(self):
        return self.revision['record']

    @property
    def geometry_revision(self):
        return self.revision['geometry_revision']
