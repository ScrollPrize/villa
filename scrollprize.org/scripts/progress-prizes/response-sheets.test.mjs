import assert from 'node:assert/strict';
import test from 'node:test';

import {
  RESPONSE_SHEET_STATIC_HEADERS,
  assertResponseSheetHeader,
  quoteSheetTitle,
  responseIdsFromValueRange,
  responseSheetHeaders,
  responseSheetQuestions,
  responseSheetRow,
} from './response-sheets.mjs';

function formFixture() {
  return {
    items: [
      {
        title: 'Your full name',
        questionItem: { question: { questionId: 'name' } },
      },
      {
        title: 'Availability',
        questionGroupItem: {
          questions: [
            { questionId: 'monday', rowQuestion: { title: 'Monday' } },
            { questionId: 'tuesday', rowQuestion: { title: 'Tuesday' } },
          ],
        },
      },
    ],
  };
}

test('derives stable response columns from ordinary and grouped questions', () => {
  const form = formFixture();
  assert.deepEqual(responseSheetQuestions(form), [
    { questionId: 'name', title: 'Your full name' },
    { questionId: 'monday', title: 'Availability — Monday' },
    { questionId: 'tuesday', title: 'Availability — Tuesday' },
  ]);
  assert.deepEqual(responseSheetHeaders(form), [
    ...RESPONSE_SHEET_STATIC_HEADERS,
    'Your full name [name]',
    'Availability — Monday [monday]',
    'Availability — Tuesday [tuesday]',
    'Raw answers JSON',
  ]);
});

test('serializes one response without interpreting formula-like answer text', () => {
  const form = formFixture();
  const row = responseSheetRow(form, {
    responseId: 'response-one',
    createTime: '2026-08-06T12:00:00Z',
    lastSubmittedTime: '2026-08-06T12:01:00Z',
    respondentEmail: 'entrant@example.org',
    answers: {
      name: { textAnswers: { answers: [{ value: '=IMPORTXML("private")' }] } },
      monday: { textAnswers: { answers: [{ value: 'Yes' }] } },
    },
  });

  assert.equal(row[0], 'response-one');
  assert.equal(row[4], '=IMPORTXML("private")');
  assert.equal(row[5], 'Yes');
  assert.equal(row[6], '');
  assert.deepEqual(JSON.parse(row.at(-1)), {
    monday: { textAnswers: { answers: [{ value: 'Yes' }] } },
    name: { textAnswers: { answers: [{ value: '=IMPORTXML("private")' }] } },
  });
});

test('requires an exact immutable header before appending response rows', () => {
  const headers = responseSheetHeaders(formFixture());
  assert.doesNotThrow(() => assertResponseSheetHeader({ values: [headers] }, headers));
  assert.throws(
    () => assertResponseSheetHeader({ values: [[...headers, 'Human-added column']] }, headers),
    /does not match/,
  );
  assert.throws(() => assertResponseSheetHeader({}, headers), /missing or malformed/);
});

test('deduplicates only stable response IDs and fails closed on existing duplicates', () => {
  assert.deepEqual(
    [...responseIdsFromValueRange({ values: [['Response ID'], ['one'], ['two'], []] })],
    ['one', 'two'],
  );
  assert.throws(
    () => responseIdsFromValueRange({ values: [['Response ID'], ['one'], ['one']] }),
    /duplicate response ID/,
  );
  assert.throws(
    () => responseIdsFromValueRange({ values: [['Wrong header']] }),
    /unexpected header/,
  );
});

test('quotes Sheet tab titles without allowing range injection', () => {
  assert.equal(quoteSheetTitle("Entrants' responses"), "'Entrants'' responses'");
});
