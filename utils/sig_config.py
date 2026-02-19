sig_configs = {
    'h_corridor': {
        'sig_ids': ['h2', 'h3', 'h4', 'h5', 'h6'],
        'left_turns': ['N-E', 'S-W'],
        'h2': {
            'phase_pairs': {
                0: ('E-S', 'W-N'),
                1: ('W-E', 'W-N'),
                2: ('W-E', 'E-W'),
                3: ('E-S', 'E-W'),
                4: ('S-W', 'N-E'),
                5: ('S-W', 'S-N'),
                6: ('S-N', 'N-S'),
                7: ('N-S', 'N-E'),
            },
            'action_phase_map': {
                0: 'grrgrGgrrgrG',  # WL EL
                1: 'grrgrrgrrgGG',  # WL WT
                2: 'grrgGrgrrgGr',  # WT ET
                3: 'grrgGGgrrgrr',  # EL ET
                4: 'grGgrrgrGgrr',  # SL NL
                5: 'grrgrrgGGgrr',  # SL ST
                6: 'gGrgrrgGrgrr',  # ST NT
                7: 'gGGgrrgrrgrr',  # NL NT
            },
            'number_lanes': 12,
            'incoming_edges': {'nh2': 0, 'w2': 3, 'sh2': 6, 'e1': 9},
            'lane_sets': {'N-W': ['nh2_0'], 'N-S': ['nh2_1'], 'N-E': ['nh2_2'], 'E-N': ['w2_0'], 'E-W': ['w2_1'],
                          'E-S': ['w2_2'], 'S-E': ['sh2_0'], 'S-N': ['sh2_1'], 'S-W': ['sh2_2'], 'W-S': ['e1_0'],
                          'W-E': ['e1_1'], 'W-N': ['e1_2']},
            'downstream': {'N': 'hn2', 'E': 'e2', 'S': 'hs2', 'W': 'w1'}},

        'h3': {
            'phase_pairs': {
                0: ('E-S', 'W-N'),
                1: ('W-E', 'W-N'),
                2: ('W-E', 'E-W'),
                3: ('E-S', 'E-W'),
                4: ('S-W', 'N-E'),
                5: ('S-W', 'S-N'),
                6: ('S-N', 'N-S'),
                7: ('N-S', 'N-E'),
            },
            'action_phase_map': {
                0: 'grrgrGgrrgrG',  # WL EL
                1: 'grrgrrgrrgGG',  # WL WT
                2: 'grrgGrgrrgGr',  # WT ET
                3: 'grrgGGgrrgrr',  # EL ET
                4: 'grGgrrgrGgrr',  # SL NL
                5: 'grrgrrgGGgrr',  # SL ST
                6: 'gGrgrrgGrgrr',  # ST NT
                7: 'gGGgrrgrrgrr',  # NL NT
            },
            'number_lanes': 12,
            'incoming_edges': {'nh3': 0, 'w3': 3, 'sh3': 6, 'e2': 9},
            'lane_sets': {'N-W': ['nh3_0'], 'N-S': ['nh3_1'], 'N-E': ['nh3_2'], 'E-N': ['w3_0'], 'E-W': ['w3_1'],
                          'E-S': ['w3_2'], 'S-E': ['sh3_0'], 'S-N': ['sh3_1'], 'S-W': ['sh3_2'], 'W-S': ['e2_0'],
                          'W-E': ['e2_1'], 'W-N': ['e2_2']},
            'downstream': {'N': 'hn3', 'E': 'e3', 'S': 'hs3', 'W': 'w2'}},

        'h4': {
            'phase_pairs': {
                0: ('E-S', 'W-N'),
                1: ('W-E', 'W-N'),
                2: ('W-E', 'E-W'),
                3: ('E-S', 'E-W'),
                4: ('S-W', 'N-E'),
                5: ('S-W', 'S-N'),
                6: ('S-N', 'N-S'),
                7: ('N-S', 'N-E'),
            },
            'action_phase_map': {
                0: 'grrgrGgrrgrG',  # WL EL
                1: 'grrgrrgrrgGG',  # WL WT
                2: 'grrgGrgrrgGr',  # WT ET
                3: 'grrgGGgrrgrr',  # EL ET
                4: 'grGgrrgrGgrr',  # SL NL
                5: 'grrgrrgGGgrr',  # SL ST
                6: 'gGrgrrgGrgrr',  # ST NT
                7: 'gGGgrrgrrgrr',  # NL NT
            },
            'number_lanes': 12,
            'incoming_edges': {'nh4': 0, 'w4': 3, 'sh4': 6, 'e3': 9},
            'lane_sets': {'N-W': ['nh4_0'], 'N-S': ['nh4_1'], 'N-E': ['nh4_2'], 'E-N': ['w4_0'], 'E-W': ['w4_1'],
                          'E-S': ['w4_2'], 'S-E': ['sh4_0'], 'S-N': ['sh4_1'], 'S-W': ['sh4_2'], 'W-S': ['e3_0'],
                          'W-E': ['e3_1'], 'W-N': ['e3_2']},
            'downstream': {'N': 'hn4', 'E': 'e4', 'S': 'hs4', 'W': 'w3'}},

        'h5': {
            'phase_pairs': {
                0: ('E-S', 'W-N'),
                1: ('W-E', 'W-N'),
                2: ('W-E', 'E-W'),
                3: ('E-S', 'E-W'),
                4: ('S-W', 'N-E'),
                5: ('S-W', 'S-N'),
                6: ('S-N', 'N-S'),
                7: ('N-S', 'N-E'),
            },
            'action_phase_map': {
                0: 'grrgrGgrrgrG',  # WL EL
                1: 'grrgrrgrrgGG',  # WL WT
                2: 'grrgGrgrrgGr',  # WT ET
                3: 'grrgGGgrrgrr',  # EL ET
                4: 'grGgrrgrGgrr',  # SL NL
                5: 'grrgrrgGGgrr',  # SL ST
                6: 'gGrgrrgGrgrr',  # ST NT
                7: 'gGGgrrgrrgrr',  # NL NT
            },
            'number_lanes': 12,
            'incoming_edges': {'nh5': 0, 'w5': 3, 'sh5': 6, 'e4': 9},
            'lane_sets': {'N-W': ['nh5_0'], 'N-S': ['nh5_1'], 'N-E': ['nh5_2'], 'E-N': ['w5_0'], 'E-W': ['w5_1'],
                          'E-S': ['w5_2'], 'S-E': ['sh5_0'], 'S-N': ['sh5_1'], 'S-W': ['sh5_2'], 'W-S': ['e4_0'],
                          'W-E': ['e4_1'], 'W-N': ['e4_2']},
            'downstream': {'N': 'hn5', 'E': 'e5', 'S': 'hs5', 'W': 'w4'}},

        'h6': {
            'phase_pairs': {
                0: ('E-S', 'W-N'),
                1: ('W-E', 'W-N'),
                2: ('W-E', 'E-W'),
                3: ('E-S', 'E-W'),
                4: ('S-W', 'N-E'),
                5: ('S-W', 'S-N'),
                6: ('S-N', 'N-S'),
                7: ('N-S', 'N-E'),
            },
            'action_phase_map': {
                0: 'grrgrGgrrgrG',  # WL EL
                1: 'grrgrrgrrgGG',  # WL WT
                2: 'grrgGrgrrgGr',  # WT ET
                3: 'grrgGGgrrgrr',  # EL ET
                4: 'grGgrrgrGgrr',  # SL NL
                5: 'grrgrrgGGgrr',  # SL ST
                6: 'gGrgrrgGrgrr',  # ST NT
                7: 'gGGgrrgrrgrr',  # NL NT
            },
            'number_lanes': 12,
            'incoming_edges': {'nh6': 0, 'w6': 3, 'sh6': 6, 'e5': 9},
            'lane_sets': {'N-W': ['nh6_0'], 'N-S': ['nh6_1'], 'N-E': ['nh6_2'], 'E-N': ['w6_0'], 'E-W': ['w6_1'],
                          'E-S': ['w6_2'], 'S-E': ['sh6_0'], 'S-N': ['sh6_1'], 'S-W': ['sh6_2'], 'W-S': ['e5_0'],
                          'W-E': ['e5_1'], 'W-N': ['e5_2']},
            'downstream': {'N': 'hn6', 'E': 'e6', 'S': 'hs6', 'W': 'w5'}},
    },
    'r_corridor': {
        'sig_ids': ['h2', 'h3', 'h4', 'h5', 'h6'],
        'left_turns': ['N-E', 'S-W'],
        'h2': {
            'phase_pairs': {
                0: ('W-E',),
                1: ('E-S', 'E-W'),
                2: ('S-W',),
            },
            'action_phase_map': {
                0: 'rrrgrgGG',  # WT
                1: 'GGggrrrr',  # EL ET
                2: 'rrrgGgrr',  # SL
            },
            'number_lanes': 5,
            'incoming_edges': {'w2': 4, 'sh2': 8, 'e1': 12},
            'lane_sets': {'E-W': ['w2_0', 'w2_1'], 'E-S': ['w2_1'],
                          'S-E': ['sh2_0'], 'S-W': ['sh2_0'],
                          'W-S': ['e1_0'], 'W-E': ['e1_0', 'e1_1']},
            'downstream': {'E': 'e2', 'S': 'hs2', 'W': 'w1'}},

        'h3': {
            'phase_pairs': {
                0: ('E-S', 'W-N'),
                1: ('W-E', 'W-N'),
                2: ('W-E', 'E-W'),
                3: ('E-S', 'E-W'),
                4: ('S-W', 'N-E'),
                5: ('S-W', 'S-N'),
                6: ('S-N', 'N-S'),
                7: ('N-S', 'N-E'),
            },
            'action_phase_map': {
                0: 'grrgrrGgrrgrrG',  # WL EL
                1: 'grrgrrrgrrgGGG',  # WL WT
                2: 'grrgGGrgrrgGGr',  # WT ET
                3: 'grrgGGGgrrgrrr',  # EL ET
                4: 'grGgrrrgrGgrrr',  # SL NL
                5: 'grrgrrrgGGgrrr',  # SL ST
                6: 'gGrgrrrgGrgrrr',  # ST NT
                7: 'gGGgrrrgrrgrrr',  # NL NT
            },
            'number_lanes': 10,
            'incoming_edges': {'nh3': 0, 'w3': 4, 'sh3': 8, 'e2': 12},
            'lane_sets': {'N-W': ['nh3_0'], 'N-S': ['nh3_0'], 'N-E': ['nh3_1'],
                          'E-N': ['w3_0'], 'E-W': ['w3_0', 'w3_1'], 'E-S': ['w3_2'],
                          'S-E': ['sh3_0'], 'S-N': ['sh3_1'], 'S-W': ['sh3_1'],
                          'W-S': ['e2_0'], 'W-E': ['e2_0', 'e2_1'], 'W-N': ['e2_2']},
            'downstream': {'N': 'hn3', 'E': 'e3', 'S': 'hs3', 'W': 'w2'}},

        'h4': {
            'phase_pairs': {
                0: ('E-S', 'W-N'),
                1: ('W-E', 'W-N'),
                2: ('W-E', 'E-W'),
                3: ('E-S', 'E-W'),
                4: ('S-W', 'N-E'),
                5: ('S-W', 'S-N'),
                6: ('S-N', 'N-S'),
                7: ('N-S', 'N-E'),
            },
            'action_phase_map': {
                0: 'grrrgrrGGgrrrgrrGG',  # WL EL
                1: 'grrrgrrrrgrrrgGGGG',  # WL WT
                2: 'grrrgGGrrgrrrgGGrr',  # WT ET
                3: 'grrrgGGGGgrrrgrrrr',  # EL ET
                4: 'grrGgrrrrgrrGgrrrr',  # SL NL
                5: 'grrrgrrrrgGGGgrrrr',  # SL ST
                6: 'gGGrgrrrrgGGrgrrrr',  # ST NT
                7: 'gGGGgrrrrgrrrgrrrr',  # NL NT
            },
            'number_lanes': 16,
            'incoming_edges': {'nh4': 0, 'w4': 4, 'sh4': 8, 'e3': 12},
            'lane_sets': {'N-W': ['nh4_0'], 'N-S': ['nh4_1', 'nh4_2'], 'N-E': ['nh4_3'],
                          'E-N': ['w4_0'], 'E-W': ['w4_0', 'w4_1'], 'E-S': ['w4_2', 'w4_3'],
                          'S-E': ['sh4_0'], 'S-N': ['sh4_1', 'sh4_2'], 'S-W': ['sh4_3'],
                          'W-S': ['e3_0'], 'W-E': ['e3_0', 'e3_1'], 'W-N': ['e3_2', 'e3_3']},
            'downstream': {'N': 'hn4', 'E': 'e4', 'S': 'hs4', 'W': 'w3'}},

        'h5': {
            'phase_pairs': {
                0: ('E-S', 'W-N'),
                1: ('W-E', 'W-N'),
                2: ('W-E', 'E-W'),
                3: ('E-S', 'E-W'),
                4: ('S-W', 'N-E'),
                5: ('S-W', 'S-N'),
                6: ('S-N', 'N-S'),
                7: ('N-S', 'N-E'),
            },
            'action_phase_map': {
                0: 'grrgrrGgrrgrrG',  # WL EL
                1: 'grrgrrrgrrgGGG',  # WL WT
                2: 'grrgGGrgrrgGGr',  # WT ET
                3: 'grrgGGGgrrgrrr',  # EL ET
                4: 'grGgrrrgrGgrrr',  # SL NL
                5: 'grrgrrrgGGgrrr',  # SL ST
                6: 'gGrgrrrgGrgrrr',  # ST NT
                7: 'gGGgrrrgrrgrrr',  # NL NT
            },
            'number_lanes': 12,
            'incoming_edges': {'nh5': 0, 'w5': 4, 'sh5': 8, 'e4': 12},
            'lane_sets': {'N-W': ['nh5_0'], 'N-S': ['nh5_1'], 'N-E': ['nh5_2'],
                          'E-N': ['w5_0'], 'E-W': ['w5_1', 'w5_2'], 'E-S': ['w5_3'],
                          'S-E': ['sh5_0'], 'S-N': ['sh5_0'], 'S-W': ['sh5_1'],
                          'W-S': ['e4_0'], 'W-E': ['e4_0', 'e4_1'], 'W-N': ['e4_2']},
            'downstream': {'N': 'hn5', 'E': 'e5', 'S': 'hs5', 'W': 'w4'}},

        'h6': {
            'phase_pairs': {
                0: ('E-S', 'W-N'),
                1: ('W-E', 'W-N'),
                2: ('W-E', 'E-W'),
                3: ('E-S', 'E-W'),
                4: ('S-W', 'N-E'),
                5: ('S-W', 'S-N'),
                6: ('S-N', 'N-S'),
                7: ('N-S', 'N-E'),
            },
            'action_phase_map': {
                0: 'grrgrrGgrrgrrG',  # WL EL
                1: 'grrgrrrgrrgGGG',  # WL WT
                2: 'grrgGGrgrrgGGr',  # WT ET
                3: 'grrgGGGgrrgrrr',  # EL ET
                4: 'grGgrrrgrGgrrr',  # SL NL
                5: 'grrgrrrgGGgrrr',  # SL ST
                6: 'gGrgrrrgGrgrrr',  # ST NT
                7: 'gGGgrrrgrrgrrr',  # NL NT
            },
            'number_lanes': 10,
            'incoming_edges': {'nh6': 0, 'w6': 4, 'sh6': 8, 'e5': 12},
            'lane_sets': {'N-W': ['nh6_0'], 'N-S': ['nh6_0'], 'N-E': ['nh6_1'],
                          'E-N': ['w6_0'], 'E-W': ['w6_0', 'w6_1'], 'E-S': ['w6_2'],
                          'S-E': ['sh6_0'], 'S-N': ['sh6_0'], 'S-W': ['sh6_1'],
                          'W-S': ['e5_0'], 'W-E': ['e5_0', 'e5_1'], 'W-N': ['e5_2']},
            'downstream': {'N': 'hn6', 'E': 'e6', 'S': 'hs6', 'W': 'w5'}},
    }
}
