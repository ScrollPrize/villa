export function bindRangeAndNumber(rangeInput, numberInput, applyValue) {
  rangeInput.addEventListener("input", () => applyValue(Number(rangeInput.value)));
  numberInput.addEventListener("input", () => {
    const raw = numberInput.value.trim();
    if (raw === "") return;
    const value = Number(raw);
    const minimum = numberInput.min === "" ? -Infinity : Number(numberInput.min);
    const maximum = numberInput.max === "" ? Infinity : Number(numberInput.max);
    if (Number.isFinite(value) && value >= minimum && value <= maximum) applyValue(value);
  });
  numberInput.addEventListener("change", () => {
    const raw = numberInput.value.trim();
    const value = raw === "" ? Number.NaN : Number(raw);
    const minimum = numberInput.min === "" ? -Infinity : Number(numberInput.min);
    const maximum = numberInput.max === "" ? Infinity : Number(numberInput.max);
    if (!Number.isFinite(value) || value < minimum || value > maximum) {
      numberInput.setCustomValidity(`Enter a finite value inside ${minimum}..${maximum}.`);
      numberInput.reportValidity();
      return;
    }
    numberInput.setCustomValidity("");
    applyValue(value);
  });
  numberInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      numberInput.blur();
    }
  });
}

export function setStatusText(statusElement, message) {
  if (statusElement.textContent === message) return;
  statusElement.textContent = message;
}

export function setSelectValue(select, value) {
  const nextValue = String(value);
  if (select.value === nextValue) return;
  select.value = nextValue;
}
