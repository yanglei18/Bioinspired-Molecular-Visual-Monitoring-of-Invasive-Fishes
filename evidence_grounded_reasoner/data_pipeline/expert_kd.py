# -*- coding: utf-8 -*-
"""
Expert knowledge base for fine-grained fish recognition
- expert_knowledge_data: {version, scoring, species[], diff_rules}
- get_fish_expert_knowledge(fish_name): returns appearance_attributes, behavior_attributes, confusions
"""

expert_knowledge_data = {
    "version": "fgvlm-reasoning-rules-v5.0-from-table-20260206",
    "scoring": {
        "weights": {"feature": 1.0, "negative": -1.5},
        "open_threshold": 2.0
    },
    "species": [
        # =========================
        # 1) Common carp
        # =========================
        {
            "name": "Common carp",
            "latin": "Cyprinus carpio",
            "aliases": ["Common carp", "Carp", "鲤鱼", "鲤"],
            "appearance_attributes": {
                "body_shape.fusiform_elongated": "lateral body elongated fusiform",
                "body_profile.slight_arch_post_head": "back slightly arched just behind the head",
                "body_color.silvergray_yellowbrown_reddishbrown": "body coloration can be silver-gray, yellow-brown, or reddish-brown",

                "dorsal_fin.single_long_base_along_back": "a single dorsal fin with a long base extending along most of the back",
                "dorsal_fin.origin_between_pectoral_pelvic": "dorsal-fin origin is located on the back region between the pectoral and pelvic fins",
                "dorsal_fin.max_height_anterior_base": "maximum dorsal-fin height occurs at the anterior part of the dorsal-fin base",

                "caudal_fin.deeply_forked": "caudal fin is deeply forked",
                "caudal_fin.lower_lobe_dark_reddish": "lower lobe of the caudal fin tends to be dark reddish",

                "scales.large_round": "scales are large and round",
                "pattern.dark_scale_margins_reticulated": "dark scale margins form a reticulated/mesh-like body pattern",
                "scales.metallic_sheen": "scales show metallic sheen",

                "head.short": "head is short",
                "snout.blunt_round": "snout is blunt and rounded",
                "barbels.paired_present": "paired barbels are present"
            },
            "behavior_attributes": {
                "swim.posture.steady_slow": "swimming is steady and slow",
                "tail_beats.wide_lateral_amplitude": "caudal peduncle shows large left–right sweeping amplitude",
                "body_wave.s_shaped_small": "body shows small-amplitude S-shaped undulation",
                "foraging.mouth_protrusion_head_down": "often protrudes the mouth while foraging; commonly feeds head-down",
                "pectoral_fin.motion_inconspicuous": "pectoral-fin movement is usually not obvious"
            },
            "negatives": {
                "barbels.absent": "no barbels",
                "dorsal_fin.short_base": "short dorsal-fin base (not extended along the back)",
                "swim.pattern_fast_straight_dominant": "fast, straight-line swimming as the dominant norm"
            },
            "confusions": [
                "Vs Crucian carp: crucian carp has a shorter and deeper oval body, no barbels, and scales without dark-edged spots; it swims more agilely and hurriedly with frequent turns.",
                "Vs Grass carp: grass carp is pale yellow-green/golden and more cylindrical, has no barbels, and its dorsal-fin base is shorter (often forming a small triangular fin when erected); it tends to swim straighter with smaller, low-frequency tail swings."
            ]
        },

        # =========================
        # 2) Crucian carp
        # =========================
        {
            "name": "Crucian carp",
            "latin": "Carassius auratus",
            "aliases": ["Crucian carp", "Crucian", "鲫鱼", "金鱼(广义)"],
            "appearance_attributes": {
                "body_shape.oval_short_deep": "lateral body is oval/elliptical, short and deep",
                "body_profile.strong_arch": "back is strongly arched",
                "body_color.silvergray_to_graygreen": "body coloration ranges from silver-gray to gray-green",

                "dorsal_fin.single_long_base": "a single dorsal fin with a long base",
                "dorsal_fin.max_height_anterior_base": "maximum dorsal-fin height occurs at the anterior part of the dorsal-fin base",

                "caudal_fin.deeply_forked": "caudal fin is deeply forked",

                "scales.small_round": "scales are small and round",
                "pattern.dark_edge_spots_absent": "dark-edged spots are not obvious and do not form a mesh-like pattern",
                "scales.metallic_sheen": "scales show metallic sheen",

                "head.small": "head is small",
                "snout.blunt_round": "snout is blunt and rounded",
                "barbels.absent": "no barbels"
            },
            "behavior_attributes": {
                "swim.posture.agile_hurried": "swims agilely and slightly hurried",
                "turning.frequent": "frequent direction changes",
                "tail_beats.wide_lateral_amplitude": "large left–right tail swings are common",
                "body_wave.s_shaped_deeper": "body shows deeper S-shaped undulation",
                "pectoral_fin.more_obvious_at_low_speed": "pectoral-fin beats can be more obvious at low speed/foraging"
            },
            "negatives": {
                "barbels.present": "barbels present",
                "pattern.reticulated_dark_scale_margins": "reticulated mesh-like scale-margin pattern as a dominant cue"
            },
            "confusions": [
                "Vs Common carp: common carp is more elongated fusiform and robust, has paired barbels, and larger scales with dark-edged reticulation; crucian carp is deeper/oval-bodied and usually smaller."
            ]
        },

        # =========================
        # 3) Mosquitofish
        # =========================
        {
            "name": "Mosquitofish",
            "latin": "Gambusia affinis",
            "aliases": ["Mosquitofish", "食蚊鱼"],
            "appearance_attributes": {
                "size.small": "small body size",
                "body_shape.fusiform_slightly_compressed": "fusiform body with slight lateral compression",
                "body_color.golden_green_or_graybrown": "body coloration is often golden-green or gray-brown",
                "head.small": "head is small",
                "snout.pointed": "snout is pointed",
                "eyes.relatively_large": "eyes are relatively large",

                "caudal_fin.unforked_short_rounded": "caudal fin is unforked, short, and rounded",
                "sexual_dimorphism.male_gonopodium": "male anal fin is modified into a gonopodium",
                "female.anal_fin_fan_shaped": "female anal fin is fan-shaped",
                "female.lateral_mark.teardrop_blotch": "females often show a distinct dark teardrop-shaped lateral blotch"
            },
            "behavior_attributes": {
                "swim.pattern.fast_nimble": "fast and nimble swimming",
                "surface_dashes.frequent": "often makes rapid surface dashes to capture prey"
            },
            "negatives": {
                "male.coloration.bright_variegated": "males with vivid variegated coloration as a dominant cue",
                "tail_fin.ornate_large": "large ornate caudal fin (guppy-like)"
            },
            "confusions": [
                "Vs Guppy: male guppies are brightly colored with complex stripes/spots on the tail and dorsal fins, while female guppies are paler; mosquitofish coloration is usually plain and females often show the teardrop lateral blotch."
            ]
        },

        # =========================
        # 4) Guppy
        # =========================
        {
            "name": "Guppy",
            "latin": "Poecilia reticulata",
            "aliases": ["Guppy", "孔雀鱼"],
            "appearance_attributes": {
                "size.small": "small body size",
                "body_shape.fusiform_elongated": "elongated fusiform body",
                "sexual_dimorphism.pronounced": "strong sexual dimorphism",

                "male.coloration.bright_variegated": "males are brightly colored with complex stripes/spots",
                "male.fins.tail_and_dorsal_ornate": "tail and dorsal fins often show ornate patterns",
                "male.anal_fin_gonopodium": "male anal fin forms a gonopodium",

                "female.coloration.duller_monochrome": "females are duller, often gray-brown/monochrome",
                "female.anal_fin_fan_shaped": "female anal fin is fan-shaped"
            },
            "behavior_attributes": {
                "swim.pattern.light_fluttering": "light, fluttering swimming",
                "tail_fin.spread_and_frequent_swings": "tail fin is often spread and swings frequently"
            },
            "negatives": {
                "female.lateral_mark.teardrop_blotch": "distinct dark teardrop lateral blotch as the dominant cue",
                "swim.pattern.rapid_surface_dashes": "rapid surface dashes as dominant pattern"
            },
            "confusions": [
                "Vs Mosquitofish: mosquitofish tends to be plain and uniform (golden-green/gray-brown); females often show a teardrop-shaped lateral blotch; guppies (especially males) are generally more colorful and ornate."
            ]
        },

        # =========================
        # 5) Grass carp
        # =========================
        {
            "name": "Grass carp",
            "latin": "Ctenopharyngodon idella",
            "aliases": ["Grass carp", "草鱼"],
            "appearance_attributes": {
                "body_shape.cylindrical_elongated": "elongated cylindrical body",
                "body_profile.back_not_prominent": "back is not prominent",
                "body_color.pale_yellowgreen_or_golden": "body coloration is pale yellow-green or golden",

                "dorsal_fin.single_short_base_triangular_when_erect": "a single dorsal fin with a short base; can appear triangular when erected",
                "dorsal_fin.max_height_mid_base": "maximum dorsal-fin height occurs around the middle of the dorsal-fin base",

                "caudal_fin.deeply_forked": "caudal fin is deeply forked",

                "scales.large_round": "scales are large and round",
                "pattern.dark_scale_margins_possible": "dark scale margins may be present (sometimes giving a subtle reticulated look)",

                "snout.pointed": "snout is pointed",
                "barbels.absent": "no barbels"
            },
            "behavior_attributes": {
                "swim.posture.straight_directional": "tends to swim straight with strong directionality",
                "tail_beats.small_amplitude_low_frequency": "small-amplitude, low-frequency tail swings",
                "body_wave.small_amplitude": "small-amplitude body undulation",
                "pectoral_fin.motion_inconspicuous": "pectoral-fin movement is usually not obvious"
            },
            "negatives": {
                "barbels.present": "barbels present",
                "dorsal_fin.long_base_along_back": "long dorsal-fin base extending along most of the back"
            },
            "confusions": [
                "Vs Common carp: common carp has a longer dorsal-fin base (nearly spanning the back) and paired barbels; grass carp has no barbels, a shorter dorsal-fin base (often triangular when erected), and swims straighter with smaller tail swings."
            ]
        },

        # =========================
        # 6) Largemouth bass
        # =========================
        {
            "name": "Largemouth bass",
            "latin": "Micropterus salmoides",
            "aliases": ["Largemouth bass", "大口黑鲈", "黑鲈"],
            "appearance_attributes": {
                "body_shape.fusiform_elongated": "elongated fusiform body",
                "body_color.dark_green": "dark green body coloration",

                "dorsal_fin.two_parts_separated": "two-part dorsal fin (spiny anterior + soft-rayed posterior) separated/not connected",
                "dorsal_fin.max_height_posterior": "maximum dorsal-fin height tends toward the rear part of the dorsal base",
                "dorsal_fin.semi_transparent": "dorsal fin is semi-transparent",

                "caudal_fin.shallow_forked": "caudal fin is shallowly forked",
                "scales.relatively_small": "scales are relatively small",

                "stripe.dark_lateral": "distinct dark lateral stripe along the body (head to tail)",
                "jaw.upper_jaw_past_eye": "upper jaw extends past the rear edge of the eye",
                "mouth.large": "large mouth"
            },
            "behavior_attributes": {
                "swim.pattern.stop_and_go_slow": "intermittent/stop-and-go swimming, often slow",
                "tail_peduncle.almost_still": "caudal peduncle is almost still",
                "dorsal_anal.rippling": "dorsal and anal fins ripple",
                "caudal_fin.vertical_oscillation": "caudal fin shows more vertical oscillation",
                "body.linearity_high": "body stays mostly straight with minimal sway",
                "pectoral_fin.motion_very_obvious": "pectoral-fin movement is very obvious"
            },
            "negatives": {
                "dorsal_fin.long_continuous_sail_like": "single long continuous sail-like dorsal fin",
                "stripe.absent": "no lateral stripe"
            },
            "confusions": [
                "Vs Mozambique tilapia: tilapia is more oval and laterally compressed, has a long continuous sail-like dorsal fin and larger, clearer scales; largemouth bass is more elongated and shows a distinct lateral stripe with a larger mouth.",
                "Vs carps (Common/Crucian): carps have a single long-based dorsal fin and strong lateral tail swings; (common carp) has barbels."
            ]
        },

        # =========================
        # 7) Mozambique tilapia
        # =========================
        {
            "name": "Mozambique tilapia",
            "latin": "Oreochromis mossambicus",
            "aliases": ["Mozambique tilapia", "Tilapia", "莫桑比克罗非鱼", "罗非鱼"],
            "appearance_attributes": {
                "body_shape.oval_laterally_compressed": "oval body, strongly laterally compressed",
                "body_color.grayblack_to_light_green": "body coloration ranges from gray-black to light green",

                "dorsal_fin.long_continuous_sail_like": "very long, continuous sail-like dorsal fin extending along the back",
                "caudal_fin.shallow_forked_or_emarginate": "caudal fin is shallowly forked/emarginate (not deeply forked)",
                "scales.large_clear_texture": "scales are larger with clearer texture",

                "snout.front_narrower": "snout/mouth front appears narrower"
            },
            "behavior_attributes": {
                "swim.pattern.stop_and_go_slow": "intermittent and often slow movement",
                "tail_peduncle.almost_still": "caudal peduncle is almost still",
                "dorsal_anal.rippling": "dorsal and anal fins ripple",
                "body.linearity_high": "body stays mostly straight with minimal sway",
                "pectoral_fin.motion_very_obvious": "pectoral-fin movement is very obvious"
            },
            "negatives": {
                "dorsal_fin.two_parts_separated": "two truly separate dorsal fins",
                "stripe.dark_lateral": "bold dark lateral stripe as a dominant cue"
            },
            "confusions": [
                "Vs Largemouth bass: bass is more elongated fusiform, typically dark green with a distinct lateral stripe and a larger mouth; tilapia is deeper/oval and has a long continuous sail-like dorsal fin with larger, clearer scales.",
                "Vs carps (Common/Crucian): carps have a single dorsal fin with a long base and stronger left–right tail swings; (common carp) has barbels."
            ]
        },

        # =========================
        # 8) Rainbow trout
        # =========================
        {
            "name": "Rainbow trout",
            "latin": "Oncorhynchus mykiss",
            "aliases": ["Rainbow trout", "虹鳟"],
            "appearance_attributes": {
                "body_shape.fusiform_elongated": "elongated fusiform body",
                "body_color.bluishgreen_or_graygreen": "body coloration is bluish-green or gray-green",
                "lateral_band.reddish_iridescent": "reddish iridescent band along the side",

                "spots.black_dense_body": "dense small black spots across the body",
                "spots.black_on_head": "black spots also appear on the head",
                "dorsal_fin.short_base_spotted": "dorsal fin is short-based and spotted",
                "adipose_fin.present": "adipose fin is present near the caudal fin",
                "caudal_fin.shallow_forked_spotted": "caudal fin is shallowly forked and spotted",

                "scales.very_small_inconspicuous": "scales are very small/inconspicuous",
                "snout.pointed": "snout is pointed"
            },
            "behavior_attributes": {
                "swim.pattern.undulating": "undulating swimming",
                "tail_beats.wide_lateral_amplitude": "large left–right tail swings",
                "body_wave.s_shaped_large": "large-amplitude S-shaped body undulation",
                "pectoral_fin.motion_obvious": "pectoral-fin movement is obvious",
                "resting.sometimes_still": "may sometimes remain motionless"
            },
            "negatives": {
                "spots.orange_yellow_conspicuous": "conspicuous orange-yellow spots as the dominant cue",
                "caudal_fin.unspotted": "unspotted caudal fin as a dominant cue"
            },
            "confusions": [
                "Vs Brown trout: brown trout is golden-brown/gray-brown with dark spots plus conspicuous orange-yellow spots; its caudal fin is typically unspotted (upper surface without spots)."
            ]
        },

        # =========================
        # 9) Brown trout
        # =========================
        {
            "name": "Brown trout",
            "latin": "Salmo trutta",
            "aliases": ["Brown trout", "褐鳟"],
            "appearance_attributes": {
                "body_shape.fusiform_elongated": "elongated fusiform body",
                "body_color.goldenbrown_or_graybrown": "body coloration is golden-brown or gray-brown",

                "spots.dark_plus_orange_yellow": "dark spots plus conspicuous orange-yellow spots on the side",
                "dorsal_fin.short_base_triangular_when_erect": "dorsal fin is short-based and can appear triangular when erected",
                "adipose_fin.present_unspotted": "adipose fin is present and typically unspotted",
                "caudal_fin.shallow_forked_unspotted": "caudal fin is shallowly forked and typically unspotted",

                "scales.very_small_inconspicuous": "scales are very small/inconspicuous",
                "head.longer_pointed": "head outline is longer with a pointed front",
                "snout.pale_graywhite": "snout often appears pale gray-white",
                "head_spots.few": "few black spots on the head"
            },
            "behavior_attributes": {
                "swim.pattern.undulating": "undulating swimming",
                "tail_beats.wide_lateral_amplitude": "large left–right tail swings",
                "body_wave.s_shaped_large": "large-amplitude S-shaped body undulation",
                "pectoral_fin.motion_obvious": "pectoral-fin movement is obvious",
                "resting.sometimes_still": "may sometimes remain motionless"
            },
            "negatives": {
                "spots.black_dense_body_and_fins": "dense black spots across body and fins (including tail/dorsal) as dominant cue",
                "caudal_fin.spotted": "spotted caudal fin as a dominant cue"
            },
            "confusions": [
                "Vs Rainbow trout: rainbow trout is bluish/gray-green with a reddish side band, and dense dark spots extend across the body and onto the head, dorsal fin, and caudal fin."
            ]
        },

        # =========================
    # 10) Redeye barbel
    # =========================
    {
        "name": "Redeye barbel",
        "latin": "Squaliobarbus curriculus",
        "aliases": ["Redeye barbel", "赤眼鳟"],

        "appearance_attributes": {
            "body_shape.slender_fusiform": "slender fusiform streamlined body",
            "body.streamlined_thin": "body relatively thin and streamlined",
            "body_profile.slight_arch_post_head": "back slightly arched behind head",
            "body_color.silvergray_yellowish": "silver-gray to yellowish-gray body",

            "eye.reddish_orbital_ring": "distinct reddish eye or orbital ring",
            "eye.yellow_patch_behind": "yellowish patch behind eye",

            "dorsal_fin.single_short_base": "single dorsal fin with short base",
            "dorsal_fin.origin.near_mid_body": "dorsal fin originates near mid-body",
            "dorsal_fin.triangular": "triangular dorsal fin",

            "caudal_fin.deeply_forked": "deeply forked caudal fin",

            "scales.medium_dark_edges": "medium-sized scales with dark edges",

            "head.small": "small head",
            "snout.blunt": "blunt snout",
            "lips.thin": "thin lips",
            "barbels.absent_or_tiny": "barbels absent or extremely small"
        },

        "behavior_attributes": {
            "swim.posture.steady_slow": "swimming steady and slow",
            "tail_swing.moderate_rhythmic": "tail swing moderate but rhythmic",
            "body_undulation.weak": "body undulation weak",
            "pectoral_fin.beats_visible_low_speed": "pectoral fin beats visible at low speed",
            "movement.delicate_than_carp": "movement more delicate than common carp"
        },

        "confusions": [
            "Vs Common carp: common carp has paired barbels and much longer dorsal-fin base.",
            "Vs Schizothorax fish: schizothorax lacks red eye markings and has thicker rough scales."
        ]
    },

    # =========================
    # 11) Mud carp
    # =========================
    {
        "name": "Mud carp",
        "latin": "Cirrhinus molitorella",
        "aliases": ["Mud carp", "桂鲮"],

        "appearance_attributes": {
            "body_shape.elongated_fusiform": "elongated fusiform body",
            "body_profile.flat_to_slight_arch": "back flat to slightly arched",
            "body_color.graygreen_olivebrown": "gray-green to olive-brown coloration",

            "dorsal_fin.single_long_base": "single dorsal fin with long base",
            "dorsal_fin.origin.above_pelvic_fins": "dorsal fin origin above pelvic fins",
            "dorsal_fin.max_height_posterior": "maximum height toward posterior base",

            "caudal_fin.forked": "forked caudal fin",

            "scales.large_round_dark_edges": "large round scales with dark edges",
            "scales.round_shape_emphasized": "scales distinctly large and round",

            "head.short": "short head",
            "head.proportion_short": "head proportionally short",
            "snout.blunt": "blunt snout",
            "mouth.terminal": "terminal mouth",
            "barbels.absent": "no barbels"
        },

        "behavior_attributes": {
            "swim.posture.steady_slow": "swimming steady and slow",
            "tail_beats.low_frequency": "low-frequency tail beats",
            "tail_beats.moderate_amplitude": "moderate amplitude tail beats",
            "caudal_peduncle.movement_obvious": "caudal peduncle movement obvious",
            "body_sway.small": "body sway small",
            "trajectory.straight": "maintains straight trajectory"
        },

        "confusions": [
            "Vs Common carp: mud carp lacks barbels and has shorter dorsal-fin base.",
            "Vs Serrated barb: serrated barb has reddish head markings and serrated scales."
        ]
    },

    # =========================
    # 12) Serrated barb
    # =========================
    {
        "name": "Serrated barb",
        "latin": "Acrossocheilus spp.",
        "aliases": ["Serrated barb", "锯齿倒刺鲃"],

        "appearance_attributes": {
            "body_shape.elongated_fusiform": "elongated fusiform body",
            "body_profile.gently_arched": "gently arched back",
            "body_color.pale_green_yellowish": "pale green to yellowish body",

            "dorsal_fin.single_moderate_base": "single dorsal fin with moderate base",
            "dorsal_fin.triangular": "triangular dorsal fin",
            "dorsal_fin.origin.behind_pectoral_fins": "origin behind pectoral fins",
            "dorsal_fin.origin.precisely_behind_pectoral": "clearly behind pectoral fins",

            "caudal_fin.forked": "forked caudal fin",

            "scales.large_serrated_edges": "large scales with serrated edges",
            "scale_edges.high_contrast_dark": "dark high-contrast serrated edges",

            "head.short": "short head",
            "snout.blunt": "blunt snout",
            "reddish_patch.cheek_head": "reddish patch on cheek or head",
            "barbels.absent": "no barbels"
        },

        "behavior_attributes": {
            "swim.posture.steady_slow": "steady slow swimming",
            "tail_swing.moderate_amplitude": "moderate tail swing",
            "caudal_fin.beats_clear": "caudal fin beats clear",
            "body_undulation.weak": "weak body undulation",
            "posture.stable_cruising": "stable cruising posture"
        },

        "confusions": [
            "Vs Mud carp: serrated barb has reddish head markings and serrated scale edges.",
            "Vs Common carp: common carp has barbels and longer dorsal-fin base."
        ]
    },

    # =========================
    # 13) Black carp
    # =========================
    {
        "name": "Black carp",
        "latin": "Mylopharyngodon piceus",
        "aliases": ["Black carp", "青鱼"],

        "appearance_attributes": {
            "body_shape.large_fusiform": "large elongated fusiform body",
            "body.build.thick_powerful": "body thick and powerful",
            "body_profile.slight_arch": "slightly arched back",
            "body_color.dark_gray_blackish": "uniform dark gray to blackish",
            "color.pattern_uniform": "uniform coloration without pattern",

            "dorsal_fin.single_long_base": "single dorsal fin with long base",
            "dorsal_fin.max_height_anterior": "maximum height toward anterior base",

            "caudal_fin.shallow_forked": "shallowly forked caudal fin",

            "scales.large_smooth": "large smooth-edged scales",

            "head.large": "large head",
            "snout.blunt": "blunt snout",
            "lips.thick": "thick lips",
            "barbels.absent": "no barbels"
        },

        "behavior_attributes": {
            "swim.posture.steady_slow": "steady slow swimming",
            "tail_swing.large_amplitude": "large amplitude tail swing",
            "caudal_fin.beats_strong": "strong caudal fin beats",
            "caudal_fin.beats_regular": "regular beats",
            "body_linearity.high": "body mostly straight",
            "lateral_sway.minimal": "minimal lateral sway"
        },

        "confusions": [
            "Vs Common carp: black carp lacks barbels and is darker uniformly.",
            "Vs Schizothorax fish: black carp has smoother larger scales."
        ]
    },

    # =========================
    # 14) Paddlefish
    # =========================
    {
        "name": "Chinese paddlefish",
        "latin": "Polyodon spathula",
        "aliases": ["Chinese paddlefish", "匙吻鲟"],

        "appearance_attributes": {
            "body_shape.extremely_elongated": "extremely elongated body",
            "rostrum.paddle_shaped": "long paddle-shaped snout",
            "skin.scaleless_smooth": "smooth scaleless skin",
            "back.flat_post_head": "back flat behind head",
            "body_color.pale_gray_silvery": "pale gray to silvery body",

            "mouth.very_large": "very large mouth",
            "dorsal_fin.small_posterior": "small posterior dorsal fin",
            "caudal_fin.heterocercal": "heterocercal tail fin",
            "head.long_flat": "long flattened head"
        },

        "behavior_attributes": {
            "swim.posture.smooth_steady": "smooth steady swimming",
            "tail_beats.slow_wide": "slow wide tail beats",
            "cruising.continuous": "continuous cruising",
            "mouth.open_frequent": "frequently open mouth while swimming",
            "feeding.filter_feeding": "filter-feeding behavior",
            "pectoral_fin.stabilizing": "pectoral fins stabilize motion"
        },

        "confusions": [
            "Vs Common carp: paddlefish lacks scales and paddle-shaped snout.",
            "Vs Sturgeon: paddlefish thinner and more flexible."
        ]
    },

    # =========================
    # 15) Wuchang bream
    # =========================
    {
        "name": "Wuchang bream",
        "latin": "Megalobrama amblycephala",
        "aliases": ["Wuchang bream", "团头鲂"],

        "appearance_attributes": {
            "body_shape.rhomboid_compressed": "rhomboid compressed body",
            "body_profile.deep_tall": "deep tall body",
            "body_color.light_bluish_gray": "light bluish-gray",

            "dorsal_fin.single_long_base": "single dorsal fin long base",
            "dorsal_fin.origin.behind_pectoral": "origin behind pectoral fins",
            "dorsal_fin.max_height_anterior_base": "max height at anterior base",

            "caudal_fin.deeply_forked": "deeply forked caudal fin",

            "scales.medium_smooth": "medium smooth scales",
            "scales.without_dark_edges": "scales without dark edges",

            "head.short": "short head",
            "snout.blunt": "blunt snout",
            "barbels.absent": "no barbels"
        },

        "behavior_attributes": {
            "swim.posture.steady_slow": "steady slow swimming",
            "tail_swing.moderate": "moderate tail swing",
            "caudal_fin.movement_clear": "clear caudal fin movement",
            "body_undulation.little": "little body undulation",
            "habitat.mid_water": "mid-water cruising"
        },

        "confusions": [
            "Vs Common carp: bream compressed rhomboid body no barbels.",
            "Vs Mud carp: mud carp fusiform not compressed."
        ]
    },

    # =========================
    # 16) Chinese sucker
    # =========================
    {
        "name": "Chinese sucker",
        "latin": "Myxocyprinus asiaticus",
        "aliases": ["Chinese sucker", "胭脂鱼"],

        "appearance_attributes": {
            "body_shape.triangular_elongated": "elongated triangular body",
            "body_color.reddishbrown_yellowish": "reddish-brown to yellowish",

            "stripe.longitudinal_dark": "dark longitudinal stripes",

            "dorsal_fin.high_sail_like": "high sail-like dorsal fin",
            "dorsal_fin.profile.high_emphasized": "extremely high dorsal fin",

            "caudal_fin.forked": "forked caudal fin",

            "scales.small_size": "small scales",
            "head.small_proportion": "small head",

            "mouth.inferior_suction": "inferior suction mouth"
        },

        "behavior_attributes": {
            "swim.bottom_oriented": "bottom-oriented swimming",
            "swim.gliding_near_substrate": "slow gliding near substrate",
            "movement.short_distance": "short distance movement",
            "posture.stable": "stable posture",
            "tail_beats.short": "short tail beats",
            "pectoral_fin.active_attachment": "pectoral fins used for attachment"
        },

        "confusions": [
            "Vs Mud carp: sucker has inferior suction mouth and benthic habit.",
            "Vs Common carp: carp mid-water with barbels."
        ]
    },

    # =========================
    # 17) Chinese labeo
    # =========================
    {
        "name": "Chinese labeo",
        "latin": "Labeo spp.",
        "aliases": ["Chinese labeo", "华鳈"],

        "appearance_attributes": {
            "body_shape.small_slender_fusiform": "small slender fusiform body",
            "back.slight_arch": "slightly arched back",
            "body_color.silverygray_darkgray": "silvery-gray to dark gray",

            "bands.vertical_yellowbrown": "vertical yellow-brown bands",

            "dorsal_fin.triangular_short_base": "triangular short-base dorsal fin",

            "caudal_fin.forked": "forked caudal fin",

            "scales.small_size": "small scales",

            "head.region_darker": "head darker than body",
            "snout.pointed": "pointed snout",
            "barbels.absent": "no barbels"
        },

        "behavior_attributes": {
            "swim.agile_active": "agile active swimming",
            "tail_beats.rapid": "rapid tail beats",
            "turning.frequent": "frequent maneuvering",
            "body_motion.flexible": "flexible body motion",
            "habitat.mid_water": "mid-water swimming common"
        },

        "confusions": [
            "Vs Chinese sucker: sucker has suction mouth and benthic behavior.",
            "Vs Redeye barbel: redeye barbel lacks vertical bands."
        ]
    },

    # =========================
    # 18) Schizothorax fish
    # =========================
    {
        "name": "Schizothorax fish",
        "latin": "Schizothorax spp.",
        "aliases": ["Schizothorax fish", "史氏新光唇"],

        "appearance_attributes": {
            "body_shape.elongated_fusiform": "elongated fusiform body",
            "body_color.grayish_olive_yellowbrown": "grayish olive to yellow-brown",

            "scales.thick_coarse_metallic": "thick coarse metallic scales",
            "scales.large_round": "large round scales",
            "scales.metallic_sheen_visible": "visible metallic sheen",

            "dorsal_fin.single_long_base": "single long-base dorsal fin",
            "dorsal_fin.base.extends_far_posterior": "dorsal-fin base extends far posteriorly",

            "caudal_fin.forked": "forked caudal fin",

            "head.short": "short head",
            "snout.blunt": "blunt snout",
            "lips.thick": "thick lips",
            "barbels.absent": "no barbels"
        },

        "behavior_attributes": {
            "swim.posture.steady_slow": "steady slow swimming",
            "tail_swing.moderate": "moderate tail swing",
            "caudal_fin.beats_regular": "regular caudal fin beats",
            "body_undulation.small": "small body undulation",
            "trajectory.straight_often": "often straight swimming"
        },

        "confusions": [
            "Vs Common carp: rough metallic scales and no barbels.",
            "Vs Mud carp: longer dorsal-fin base with metallic sheen."
        ]
    }
    ],
}


def get_fish_expert_knowledge(fish_name):
    """获取指定鱼类的专家知识（按名称或别名匹配）

    Args:
        fish_name (str): 鱼类名称或别名（大小写不敏感）

    Returns:
        dict: 包含 appearance_attributes / behavior_attributes / confusions 的字典
    """
    fish_name_lower = str(fish_name).strip().lower() if fish_name is not None else ""
    for fish in expert_knowledge_data.get("species", []):
        if fish_name_lower == str(fish.get("name", "")).lower():
            return {
                "appearance_attributes": fish.get("appearance_attributes", {}),
                "behavior_attributes": fish.get("behavior_attributes", {}),
                "confusions": fish.get("confusions", [])
            }
        for alias in fish.get("aliases", []):
            if fish_name_lower == str(alias).lower():
                return {
                    "appearance_attributes": fish.get("appearance_attributes", {}),
                    "behavior_attributes": fish.get("behavior_attributes", {}),
                    "confusions": fish.get("confusions", [])
                }
    return {"appearance_attributes": {}, "behavior_attributes": {}, "confusions": []}


if __name__ == "__main__":
    # quick sanity check
    for q in ["Common carp", "Common carp", "Mosquitofish", "Guppy", "Grass carp", "Largemouth bass", "Mozambique tilapia", "Rainbow trout", "Brown trout", "Redeye barbel", "Mud carp", "Serrated barb", "Black carp", "Chinese paddlefish", "Wuchang bream", "Chinese sucker", "Chinese labeo", "Schizothorax fish"]:
        info = get_fish_expert_knowledge(q)
        print("=" * 80)
        print("Query:", q)
        print("Appearance keys:", len(info["appearance_attributes"]))
        print("Behavior keys:", len(info["behavior_attributes"]))
        print("Confusions:", info["confusions"])
