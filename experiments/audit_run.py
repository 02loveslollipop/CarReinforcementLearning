"""Audit a completed evolutionary run against its saved evidence."""
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.orchestrate import _select_parents


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(data)
    return hasher.digest()


def audit(run_dir: Path):
    state = json.loads((run_dir / 'state.json').read_text())
    problems = []
    generations = state['generations_data']
    starts = {(track, fraction) for track in ('circuit', 'oval', 'clover') for fraction in (0, .25, .5, .75)}
    immutable = run_dir / 'immutable'
    fixed = ['dqn/evaluate.py'] + [str(p.relative_to(immutable)) for p in (immutable / 'car_rl').glob('*.py')]
    fixed_hashes = {path: digest(immutable / path) for path in fixed}
    if len(generations) != state['generations']:
        problems.append('Incorrect generation count')
    all_scores = []
    for g in range(state['generations']):
        generation = generations.get(str(g))
        if generation is None or len(generation['candidates']) != state['population']:
            problems.append(f'Generation {g + 1}: missing candidates')
            continue
        candidates = generation['candidates']
        if g:
            prior = generations[str(g - 1)]['candidates']
            count = state.get('parent_count', 2)
            seed = state['seed'] + g
        elif state.get('parent_run'):
            source = json.loads((Path(state['parent_run']) / 'state.json').read_text())
            source_generation = generation['source_generation']
            prior = source['generations_data'][str(source_generation)]['candidates']
            count = math.ceil(len(prior) * .2)
            seed = state['seed'] + source_generation + 1
        else:
            prior, count, seed = [], 0, 0
        selected = generation.get('selected_parents', [])
        if count and selected != [p['index'] for p in _select_parents(prior, seed, count)]:
            problems.append(f'Generation {g + 1}: parent selection mismatch')
        for c in candidates:
            label = f"generation {g + 1}, candidate {c['index']}"
            directory = Path(c['directory'])
            if any(c.get(key) != 'complete' for key in ('status', 'agent_status', 'training_status')):
                problems.append(f'{label}: incomplete')
                continue
            try:
                for path, expected in fixed_hashes.items():
                    if digest(directory / path) != expected:
                        problems.append(f'{label}: changed fixed code {path}')
                report = json.loads((directory / 'evaluation.json').read_text())
                results = report['results']
                if len(results) != 12 or {(r['track'], r['start_fraction']) for r in results} != starts:
                    problems.append(f'{label}: incorrect benchmark starts')
                for result in results:
                    expected = (.8 * min(1, max(0, result['progress']))
                                + .2 * min(1, max(0, result['mean_forward_speed_mps'] / 20)))
                    if not math.isclose(result['score'], expected, abs_tol=1e-10):
                        problems.append(f'{label}: inconsistent episode score')
                        break
                score = float(c['score'])
                if (not math.isfinite(score) or not math.isclose(score, report['mean_score'], abs_tol=1e-10)
                        or not math.isclose(score, sum(r['score'] for r in results) / len(results), abs_tol=1e-10)):
                    problems.append(f'{label}: inconsistent score')
                all_scores.append(score)
                log = (directory / 'logs' / f"agent_attempt_{c['attempt']}.log").read_text(errors='replace')
                if 'model: gpt-6-luna' not in log or 'reasoning effort: high' not in log:
                    problems.append(f'{label}: agent settings not recorded')
                if not (directory / 'strategy.md').is_file() or not (directory / 'output/latest.pt').is_file():
                    problems.append(f'{label}: missing strategy or checkpoint')
                if count:
                    parent_index = c['parent_index']
                    if parent_index not in selected:
                        problems.append(f'{label}: parent not selected')
                    else:
                        parent = next(p for p in prior if p['index'] == parent_index)
                        if digest(directory / 'parent.pt') != digest(Path(parent['checkpoint'])):
                            problems.append(f'{label}: parent checkpoint mismatch')
                    if "Previous generation's selected winner reports:" not in log:
                        problems.append(f'{label}: selected strategy context missing')
            except (OSError, KeyError, ValueError, StopIteration, ZeroDivisionError) as error:
                problems.append(f'{label}: {error}')
    best = state.get('best')
    if not best or (all_scores and best['score'] < max(all_scores)):
        problems.append('Best record is missing or not the highest score')
    elif digest(run_dir / 'best/latest.pt') != digest(Path(best['directory']) / 'output/latest.pt'):
        problems.append('Best checkpoint does not match its source')
    if best:
        best_report = json.loads((run_dir / 'best/evaluation.json').read_text())
        if not math.isclose(best['score'], best_report['mean_score'], abs_tol=1e-10):
            problems.append('Best score does not match its saved evaluation')
    return problems


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    issues = audit(args.run_dir.resolve())
    if issues:
        print('\n'.join(issues))
        raise SystemExit(1)
    print('Audit passed: all candidates, parent handoffs, and benchmark scores verified.')
