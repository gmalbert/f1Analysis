export type JsonRow = Record<string, any>;
export type JsonObject = Record<string, any>;

export interface ApiClient {
  get<T extends JsonObject = JsonObject>(url: string): Promise<T>;
  post<T extends JsonObject = JsonObject>(url: string, body: unknown): Promise<T>;
}

export const api: ApiClient;
export function downloadUrl(path: string): string;