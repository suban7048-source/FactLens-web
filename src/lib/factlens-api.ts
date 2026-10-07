import { queryOptions } from '@tanstack/react-query';

const API = 'https://fact-lens-web.vercel.app';
export type Prediction = { label: 'REAL' | 'FAKE'; confidence: number; model: string; explanation: string };
export type ModelScore = { accuracy: number; precision: number; recall: number; f1_score: number; confusion_matrix: number[][] };
export type Metrics = { best_model: string; dataset_size: number; test_size: number; models: Record<string, ModelScore> };
async function readResponse(response: Response) {
  if (!response.ok) throw new Error('The analysis service is unavailable. Please try again shortly.');
  return response.json();
}
export async function predict(text: string): Promise<Prediction> {
  const result = await readResponse(await fetch(`${API}/api/predict`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }), signal: AbortSignal.timeout(25000) }));
  if (!['REAL', 'FAKE'].includes(result.label) || typeof result.confidence !== 'number' || result.confidence < 0 || result.confidence > 1) throw new Error('The service returned an unexpected result. Please try again.');
  return result;
}
export const metricsOptions = queryOptions({ queryKey: ['factlens-metrics'], queryFn: async (): Promise<Metrics | null> => {
  try { const data = await readResponse(await fetch(`${API}/api/metrics`, { signal: AbortSignal.timeout(8000) })); return data.models ? data : null; } catch { return null; }
}, staleTime: 300000 });