import { describe, it, expect } from 'vitest';
import {
  cleanExtractedText,
  detectSectionHeading,
  splitTextRecursively,
  computeDeterministicEmbedding,
  cosineSimilarity,
  RAGEngine,
} from '../ragEngine.js';

describe('Enso HR Policy Assistant — RAG Engine Test Suite', () => {
  it('1. Cleans extracted PDF text while preserving policy numbers and headings', () => {
    const raw = 'HR Manual - 60\n[G] Leave Policy\ncalculat - ed for 12 days( - 90days)';
    const cleaned = cleanExtractedText(raw);
    expect(cleaned).toContain('calculated');
    expect(cleaned).toContain('12 days');
    expect(cleaned).toContain('(90 days)');
    expect(cleaned).toContain('[G] Leave Policy');
  });

  it('2. Detects policy section headings accurately', () => {
    const probText = 'HR Manual - 42\n[B] Probation Policy\nMeaning/Purpose';
    expect(detectSectionHeading(probText)).toBe('[B] Probation Policy');

    const matText = 'HR Manual - 62\nLeave Type Eligibility\nMaternity Leave (Leeway)\n12 weeks (90 days)';
    expect(detectSectionHeading(matText, '[G] Leave Policy')).toBe('[G] Leave Policy — Maternity Leave (Leeway)');
  });

  it('3. Splits text recursively with overlap without losing content', () => {
    const longText = Array.from({ length: 20 }, (_, i) => `Paragraph ${i + 1}: Policy clause detail for employees.`).join('\n\n');
    const chunks = splitTextRecursively(longText, 220, 40);
    expect(chunks.length).toBeGreaterThan(2);
    expect(chunks[0]).toContain('Paragraph 1');
    expect(chunks[chunks.length - 1]).toContain('Paragraph 20');
  });

  it('4. Produces matching embedding dimensions for queries and documents', () => {
    const docVec = computeDeterministicEmbedding('Privilege Leave (PL) – 12 days');
    const queryVec = computeDeterministicEmbedding('How many days of Privilege Leave?');
    expect(docVec.length).toBe(256);
    expect(queryVec.length).toBe(256);
    expect(cosineSimilarity(docVec, queryVec)).toBeGreaterThan(0.2);
  });

  it('5. Rejects fabricated chunk citations and falls back to exact Not found', () => {
    const engine = new RAGEngine();
    const fakeRetrieved = [
      {
        chunk: {
          chunk_id: 'hr_manual_p060_c01',
          source: 'hr_manual.pdf',
          page: 60,
          section: '[G] Leave Policy',
          text: 'Privilege Leave (PL) – 12 days.',
          metadata: {},
        },
        score: 0.85,
      },
    ];

    const fabricated = JSON.stringify({
      answer: '30 days of leave',
      found: true,
      cited_chunk_ids: ['non_existent_chunk_999'],
      message: 'Fake citation',
    });

    const contract = engine.validateAndBuildContract(fabricated, fakeRetrieved);
    expect(contract.found).toBe(false);
    expect(contract.answer).toBe('Not found');
    expect(contract.sources).toEqual([]);
  });

  it('6. Safely handles malformed LLM output by returning Not found', () => {
    const engine = new RAGEngine();
    const contract = engine.validateAndBuildContract('NOT_JSON {{{', []);
    expect(contract.found).toBe(false);
    expect(contract.answer).toBe('Not found');
    expect(contract.sources).toEqual([]);
  });

  it('7. Evaluates real HR manual questions and returns verified citations or Not found', async () => {
    const engine = new RAGEngine({ disableLiveLlm: true });
    await engine.ensureInitialized();

    const q1 = await engine.answerQuestion('How many days of Privilege Leave (PL) does a permanent employee get per year?');
    expect(q1.found).toBe(true);
    expect(q1.answer).toContain('12 days');
    expect(q1.sources[0].page).toBe(60);

    const q2 = await engine.answerQuestion('What is the probation period for employees up to Senior Manager?');
    expect(q2.found).toBe(true);
    expect(q2.answer).toContain('3 months');
    expect(q2.sources[0].page).toBe(42);

    const q3 = await engine.answerQuestion("What is the company's health-insurance premium amount?");
    expect(q3.found).toBe(false);
    expect(q3.answer).toBe('Not found');
    expect(q3.sources).toEqual([]);

    const q4 = await engine.answerQuestion('What is the current GST rate in India?');
    expect(q4.found).toBe(false);
    expect(q4.answer).toBe('Not found');
    expect(q4.sources).toEqual([]);

    const q5 = await engine.answerQuestion('How many days of maternity leave are given?');
    expect(q5.found).toBe(true);
    expect(q5.answer).toMatch(/90 days|12 weeks/i);
    expect(q5.sources[0].page).toBe(62);
  });
});
