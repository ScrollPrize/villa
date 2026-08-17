export const RESPONSE_SHEET_MIME_TYPE = 'application/vnd.google-apps.spreadsheet';
export const RESPONSE_SHEET_ROLE = 'responses';
export const RESPONSE_SHEET_TAB_FALLBACK = 'Sheet1';
export const RESPONSE_SHEET_STATIC_HEADERS = Object.freeze([
  'Response ID',
  'Created at',
  'Last submitted at',
  'Respondent email',
]);

function assertNonEmptyString(value, label) {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new TypeError(`${label} must be a non-empty string`);
  }
  return value;
}

function canonicalJson(value) {
  if (Array.isArray(value)) return value.map(canonicalJson);
  if (value === null || typeof value !== 'object') return value;
  return Object.fromEntries(
    Object.entries(value)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, child]) => [key, canonicalJson(child)]),
  );
}

function collectQuestions(items, result = [], parentTitle = '') {
  for (const item of items ?? []) {
    if (item === null || typeof item !== 'object') continue;
    const itemTitle = typeof item.title === 'string' && item.title.trim() !== ''
      ? item.title.trim()
      : parentTitle;
    const question = item.questionItem?.question;
    if (typeof question?.questionId === 'string' && question.questionId !== '') {
      result.push(Object.freeze({
        questionId: question.questionId,
        title: itemTitle || 'Untitled question',
      }));
    }
    for (const groupedQuestion of item.questionGroupItem?.questions ?? []) {
      if (
        groupedQuestion
        && typeof groupedQuestion.questionId === 'string'
        && groupedQuestion.questionId !== ''
      ) {
        const rowTitle = typeof groupedQuestion.rowQuestion?.title === 'string'
          ? groupedQuestion.rowQuestion.title.trim()
          : '';
        result.push(Object.freeze({
          questionId: groupedQuestion.questionId,
          title: [itemTitle, rowTitle].filter(Boolean).join(' — ') || 'Untitled question',
        }));
      }
    }
    if (Array.isArray(item.pageBreakItem?.items)) {
      collectQuestions(item.pageBreakItem.items, result, itemTitle);
    }
  }
  return result;
}

export function responseSheetQuestions(form) {
  const questions = collectQuestions(form?.items);
  const ids = new Set();
  for (const question of questions) {
    if (ids.has(question.questionId)) {
      throw new Error('Form structure contains a duplicate question identifier');
    }
    ids.add(question.questionId);
  }
  return Object.freeze(questions);
}

export function responseSheetHeaders(form) {
  const questionHeaders = responseSheetQuestions(form).map(
    ({ questionId, title }) => `${title} [${questionId}]`,
  );
  return Object.freeze([
    ...RESPONSE_SHEET_STATIC_HEADERS,
    ...questionHeaders,
    'Raw answers JSON',
  ]);
}

function readableAnswer(answer) {
  const text = answer?.textAnswers?.answers;
  if (Array.isArray(text)) {
    return text.map((entry) => entry?.value ?? '').join('\n');
  }
  const files = answer?.fileUploadAnswers?.answers;
  if (Array.isArray(files)) {
    return files.map((entry) => entry?.fileName ?? '').join('\n');
  }
  return answer === undefined ? '' : JSON.stringify(canonicalJson(answer));
}

export function responseSheetRow(form, response) {
  if (response === null || typeof response !== 'object' || Array.isArray(response)) {
    throw new TypeError('response must be an object');
  }
  const responseId = assertNonEmptyString(response.responseId, 'response.responseId');
  const createTime = assertNonEmptyString(response.createTime, 'response.createTime');
  const lastSubmittedTime = assertNonEmptyString(
    response.lastSubmittedTime,
    'response.lastSubmittedTime',
  );
  const answers = response.answers ?? {};
  if (answers === null || typeof answers !== 'object' || Array.isArray(answers)) {
    throw new TypeError('response.answers must be an object when supplied');
  }
  return Object.freeze([
    responseId,
    createTime,
    lastSubmittedTime,
    typeof response.respondentEmail === 'string' ? response.respondentEmail : '',
    ...responseSheetQuestions(form).map(
      ({ questionId }) => readableAnswer(answers[questionId]),
    ),
    JSON.stringify(canonicalJson(answers)),
  ]);
}

export function assertResponseSheetHeader(valueRange, expectedHeaders) {
  if (!Array.isArray(expectedHeaders) || expectedHeaders.length === 0) {
    throw new TypeError('expectedHeaders must be a non-empty array');
  }
  const values = valueRange?.values;
  if (!Array.isArray(values) || values.length !== 1 || !Array.isArray(values[0])) {
    throw new Error('Managed response Sheet header is missing or malformed');
  }
  if (JSON.stringify(values[0]) !== JSON.stringify(expectedHeaders)) {
    throw new Error('Managed response Sheet header does not match the form structure');
  }
}

export function responseIdsFromValueRange(valueRange) {
  const values = valueRange?.values;
  if (values === undefined) return new Set();
  if (!Array.isArray(values) || values.some((row) => !Array.isArray(row))) {
    throw new Error('Managed response Sheet ID column is malformed');
  }
  const ids = new Set();
  for (const [index, row] of values.entries()) {
    if (index === 0) {
      if (row[0] !== RESPONSE_SHEET_STATIC_HEADERS[0]) {
        throw new Error('Managed response Sheet ID column has an unexpected header');
      }
      continue;
    }
    if (row.length === 0 || row[0] === '') continue;
    if (typeof row[0] !== 'string' || ids.has(row[0])) {
      throw new Error('Managed response Sheet contains an invalid or duplicate response ID');
    }
    ids.add(row[0]);
  }
  return ids;
}

export function quoteSheetTitle(title) {
  return `'${assertNonEmptyString(title, 'sheet title').replaceAll("'", "''")}'`;
}
