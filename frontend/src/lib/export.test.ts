import { toCsv } from "./export";

describe("toCsv", () => {
  it("usa as chaves da primeira linha e escapa vírgulas, aspas e quebras de linha", () => {
    const csv = toCsv([
      { a: 1, b: 'x,"y"', c: null },
      { a: 2, b: "linha\nnova", c: ["p", "q"] },
    ]);
    expect(csv).toBe('a,b,c\n1,"x,""y""",\n2,"linha\nnova",p|q');
  });
  it("respeita a lista de colunas e devolve vazio sem linhas", () => {
    expect(toCsv([{ a: 1, b: 2 }], ["b"])).toBe("b\n2");
    expect(toCsv([])).toBe("");
  });
});
