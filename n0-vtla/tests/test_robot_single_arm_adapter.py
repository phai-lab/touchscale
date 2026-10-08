import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from robot_single_arm_adapter import causal_indices, contiguous_runs, decode_commands, pack_commands


def test_causal_boundaries_and_stale():
    idx, age, valid = causal_indices([10,20,20,40], [9,10,19,20,39,40,55],10)
    assert idx.tolist() == [0,0,0,2,2,3,3]
    assert valid.tolist() == [False,True,True,True,False,True,False]
    assert np.all(age[valid] >= 0)
    with unittest.TestCase().assertRaises(ValueError):
        causal_indices([20,10],[20],10)


def test_runs_do_not_join_gaps():
    valid = np.r_[np.ones(49),0,np.ones(50),0,np.ones(51)].astype(bool)
    runs = contiguous_runs(valid)
    assert [len(x) for x in runs] == [50,51]
    assert runs[0][0] == 50


def test_motor_and_rotation_roundtrip():
    arm = np.array([[285,-50,150,127.285725,.009053,127.272719],[350,80,210,0,0,0]])
    hand = np.array([[0,200,400,600,800,1000],[17,23,39,55,76,99]])
    packed = pack_commands(arm,hand)
    decoded, motors = decode_commands(packed)
    assert packed.shape == (2,32)
    np.testing.assert_allclose(decoded[:,:3],arm[:,:3],atol=2e-5)
    np.testing.assert_array_equal(motors,hand)
    np.testing.assert_allclose(Rotation.from_rotvec(np.deg2rad(decoded[:,3:])).as_matrix(),
                               Rotation.from_rotvec(np.deg2rad(arm[:,3:])).as_matrix(),atol=1e-6)
    assert not packed[:,:10].any() and not packed[:,26:].any()


def test_invalid_commands_rejected():
    with unittest.TestCase().assertRaises(ValueError):
        pack_commands(np.full((1,6),np.nan),np.zeros((1,6)))
    with unittest.TestCase().assertRaises(ValueError):
        pack_commands(np.zeros((1,6)),np.full((1,6),1001))


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for fn in (
        test_causal_boundaries_and_stale, test_runs_do_not_join_gaps,
        test_motor_and_rotation_roundtrip, test_invalid_commands_rejected,
    ))


if __name__ == "__main__":
    unittest.main()
