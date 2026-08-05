import situational as sit

def check(name, cases):
    ok = True
    for value, expected, fn in cases:
        got = fn(value)
        mark = "OK" if got == expected else "FAIL"
        if got != expected:
            ok = False
        print(f"  [{mark}] {name}({value}) = {got!r}  (expected {expected!r})")
    return ok

all_ok = True

all_ok &= check("yards_gained", [
    (-5, "loss", lambda v: sit.bucket_edges(v, sit.YARDS_GAINED_EDGES, sit.YARDS_GAINED_BUCKETS)),
    (-1, "loss", lambda v: sit.bucket_edges(v, sit.YARDS_GAINED_EDGES, sit.YARDS_GAINED_BUCKETS)),
    (0,  "no_gain", lambda v: sit.bucket_edges(v, sit.YARDS_GAINED_EDGES, sit.YARDS_GAINED_BUCKETS)),
    (1,  "short_1-3", lambda v: sit.bucket_edges(v, sit.YARDS_GAINED_EDGES, sit.YARDS_GAINED_BUCKETS)),
    (3,  "short_1-3", lambda v: sit.bucket_edges(v, sit.YARDS_GAINED_EDGES, sit.YARDS_GAINED_BUCKETS)),
    (4,  "medium_4-6", lambda v: sit.bucket_edges(v, sit.YARDS_GAINED_EDGES, sit.YARDS_GAINED_BUCKETS)),
    (6,  "medium_4-6", lambda v: sit.bucket_edges(v, sit.YARDS_GAINED_EDGES, sit.YARDS_GAINED_BUCKETS)),
    (7,  "good_7-10", lambda v: sit.bucket_edges(v, sit.YARDS_GAINED_EDGES, sit.YARDS_GAINED_BUCKETS)),
    (10, "good_7-10", lambda v: sit.bucket_edges(v, sit.YARDS_GAINED_EDGES, sit.YARDS_GAINED_BUCKETS)),
    (11, "big_11-20", lambda v: sit.bucket_edges(v, sit.YARDS_GAINED_EDGES, sit.YARDS_GAINED_BUCKETS)),
    (20, "big_11-20", lambda v: sit.bucket_edges(v, sit.YARDS_GAINED_EDGES, sit.YARDS_GAINED_BUCKETS)),
    (21, "explosive_20+", lambda v: sit.bucket_edges(v, sit.YARDS_GAINED_EDGES, sit.YARDS_GAINED_BUCKETS)),
])

all_ok &= check("score_diff", [
    (-17, "down_17+", sit.bucket_score_diff),
    (-16, "down_9-16", sit.bucket_score_diff),
    (-9,  "down_9-16", sit.bucket_score_diff),
    (-8,  "down_3-8", sit.bucket_score_diff),
    (-3,  "down_3-8", sit.bucket_score_diff),
    (-2,  "close", sit.bucket_score_diff),
    (0,   "close", sit.bucket_score_diff),
    (2,   "close", sit.bucket_score_diff),
    (3,   "up_3-8", sit.bucket_score_diff),
    (8,   "up_3-8", sit.bucket_score_diff),
    (9,   "up_9-16", sit.bucket_score_diff),
    (16,  "up_9-16", sit.bucket_score_diff),
    (17,  "up_17+", sit.bucket_score_diff),
])

all_ok &= check("return_yards", [
    (0,  "none", lambda v: sit.bucket_edges(v, sit.RETURN_YARDS_EDGES, sit.RETURN_YARDS_BUCKETS)),
    (1,  "short_1-5", lambda v: sit.bucket_edges(v, sit.RETURN_YARDS_EDGES, sit.RETURN_YARDS_BUCKETS)),
    (5,  "short_1-5", lambda v: sit.bucket_edges(v, sit.RETURN_YARDS_EDGES, sit.RETURN_YARDS_BUCKETS)),
    (6,  "medium_6-15", lambda v: sit.bucket_edges(v, sit.RETURN_YARDS_EDGES, sit.RETURN_YARDS_BUCKETS)),
    (15, "medium_6-15", lambda v: sit.bucket_edges(v, sit.RETURN_YARDS_EDGES, sit.RETURN_YARDS_BUCKETS)),
    (16, "long_16-30", lambda v: sit.bucket_edges(v, sit.RETURN_YARDS_EDGES, sit.RETURN_YARDS_BUCKETS)),
    (30, "long_16-30", lambda v: sit.bucket_edges(v, sit.RETURN_YARDS_EDGES, sit.RETURN_YARDS_BUCKETS)),
    (31, "explosive_30+", lambda v: sit.bucket_edges(v, sit.RETURN_YARDS_EDGES, sit.RETURN_YARDS_BUCKETS)),
])

all_ok &= check("distance", [
    ("1", "1-3", lambda d: sit.bucket_distance(1, "1")),
    ("3", "1-3", lambda d: sit.bucket_distance(3, "1")),
    ("4", "4-6", lambda d: sit.bucket_distance(4, "1")),
    ("10", "10", lambda d: sit.bucket_distance(10, "1")),
    ("11", "11+", lambda d: sit.bucket_distance(11, "1")),
])

all_ok &= check("field_zone", [
    (95, "own_deep", sit.bucket_field_zone),
    (80, "own_deep", sit.bucket_field_zone),
    (79, "own_territory", sit.bucket_field_zone),
    (50, "own_territory", sit.bucket_field_zone),
    (49, "midfield", sit.bucket_field_zone),
    (40, "midfield", sit.bucket_field_zone),
    (39, "opp_territory", sit.bucket_field_zone),
    (20, "opp_territory", sit.bucket_field_zone),
    (19, "red_zone", sit.bucket_field_zone),
    (1,  "red_zone", sit.bucket_field_zone),
])

print()
print("ALL PASS" if all_ok else "SOME FAILED")
