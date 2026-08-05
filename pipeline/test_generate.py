from dataclasses import replace

import torch
import pytest

from dataset import PlayDataset
from generate import GameSimulator, GameState
from model import GameTransformer


@pytest.fixture(scope="module")
def small_dataset():
    return PlayDataset(history_seasons=[2022], training_seasons=[2023], data_dir="../data", max_examples=1500)


@pytest.fixture(scope="module")
def simulator(small_dataset):
    torch.manual_seed(0)
    model = GameTransformer(small_dataset.vocabs, block_size=16, n_embd=32, n_head=2, n_layer=2, dropout=0.0)
    model.eval()
    seed_game_id = small_dataset.examples[0]["game_id"]
    return GameSimulator(model, small_dataset, seed_game_id, device="cpu")


def _initial_state(sim):
    return GameState(
        quarter=1, play_in_quarter=0, down=1, ydstogo=10, yardline_100=75,
        posteam=sim.team_a, defteam=sim.team_b, posteam_score=0, defteam_score=0,
    )


def test_generate_runs_without_crashing_and_holds_invariants(simulator):
    generator = torch.Generator().manual_seed(1)
    log = simulator.generate(60, _initial_state(simulator), generator=generator)

    assert len(log) > 0
    for entry in log:
        s = entry["state"]
        assert 1 <= s.down <= 4
        assert 0 <= s.yardline_100 <= 100
        assert 1 <= s.ydstogo <= 100
        assert 1 <= s.quarter <= 5
        assert s.posteam_score >= 0 and s.defteam_score >= 0
        assert s.posteam != s.defteam
        assert {s.posteam, s.defteam} == {simulator.team_a, simulator.team_b}


def test_generate_stops_after_four_quarters(simulator):
    generator = torch.Generator().manual_seed(2)
    # far more plays than a real game has -- generate() must stop itself via
    # the quarter > 4 check rather than running forever.
    log = simulator.generate(2000, _initial_state(simulator), generator=generator)
    assert all(entry["state"].quarter <= 4 for entry in log)
    assert len(log) < 2000


def test_fourth_down_always_punts(simulator):
    generator = torch.Generator().manual_seed(3)
    state = replace(_initial_state(simulator), down=4)
    log = simulator.generate(1, state, generator=generator)
    assert log[0]["event"] == "punt"
    assert log[0]["state"].down == 1
    assert log[0]["state"].posteam == simulator.team_b  # possession flipped


def test_touchdown_adds_seven_and_flips_possession_via_kickoff(simulator):
    # run several short rollouts and require at least one touchdown to show up
    # somewhere across them, then check its scoring/possession bookkeeping
    # against the immediately preceding state.
    for seed in range(10):
        initial = _initial_state(simulator)
        log = simulator.generate(30, initial, generator=torch.Generator().manual_seed(seed))
        prev_state = initial
        for entry in log:
            if entry["event"] == "touchdown":
                s = entry["state"]
                # the team that just scored is now the DEFENSE (post-kickoff,
                # possession flipped to the team that got scored on)
                assert s.defteam_score == prev_state.posteam_score + 7
                assert s.posteam_score == prev_state.defteam_score
                assert s.down == 1 and s.ydstogo == 10
                return
            prev_state = entry["state"]
    pytest.fail("expected at least one touchdown across sampled rollouts")
