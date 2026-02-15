import json
import numpy as np
import pandas as pd
from json_helpers import json_default, safe_dump


def test_json_default_numpy_types():
    payload = {
        'nbool': np.bool_(True),
        'nint': np.int64(123),
        'nfloat': np.float64(3.14),
        'narray': np.array([1, 2, 3]),
        'pdts': pd.Timestamp('2021-07-01T12:34:56')
    }

    s = json.dumps(payload, default=json_default)
    obj = json.loads(s)

    assert obj['nbool'] is True
    assert obj['nint'] == 123
    assert abs(obj['nfloat'] - 3.14) < 1e-9
    assert obj['narray'] == [1, 2, 3]
    assert obj['pdts'].startswith('2021-07-01T12:34:56')


def test_safe_dump_writes_file(tmp_path):
    payload = {'flag': np.bool_(False), 'arr': np.array([4, 5])}
    out = tmp_path / 'out.json'
    safe_dump(payload, out, indent=2)

    assert out.exists()
    data = json.loads(out.read_text(encoding='utf-8'))
    assert data['flag'] is False
    assert data['arr'] == [4, 5]
