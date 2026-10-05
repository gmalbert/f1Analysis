import {describe, expect, it} from 'vitest';
import {readPresets, readSharedView, safeValues, savePreset, shareUrl} from './preferences';

describe('private-safe analysis settings', () => {
  it('restores Unicode values while excluding uploads, financial inputs, and malformed primitives', () => {
    const values = {filter_driver:['Émile 🏎️'], range_filter_grandPrixYear:[2017,2026],
      filter_results_main:false, filter_csv:'private', filter_token:'secret',
      bet_amount:50, upload:{content:'private'}, filter_invalid:Infinity, filter_large:'a'.repeat(4097)};
    expect(safeValues(values)).toEqual({filter_driver:['Émile 🏎️'], range_filter_grandPrixYear:[2017,2026], filter_results_main:false});
    const link = new URL(shareUrl(2,values,'http://localhost/'));
    expect(readSharedView(link.hash)).toEqual({version:1,page:2,values:safeValues(values)});
    expect(safeValues(null)).toEqual({});
    expect(safeValues('not an object')).toEqual({});
  });

  it('replaces names, limits saved entries, and handles unavailable/corrupt storage', () => {
    const values = new Map();
    const storage = {getItem:key => values.get(key), setItem:(key,value) => values.set(key,value)};
    for(let index=0;index<25;index++) savePreset('View '+index,1,{},storage);
    expect(readPresets(storage)).toHaveLength(20);
    savePreset('View 24',2,{filter_results_main:false},storage);
    expect(readPresets(storage)[0].page).toBe(2);
    expect(readPresets({getItem:() => {throw new Error('Storage denied');}})).toEqual([]);
    expect(readPresets({getItem:() => '{}'})).toEqual([]);
    expect(() => readSharedView('#/?view=invalid%')).toThrow();
  });
});
