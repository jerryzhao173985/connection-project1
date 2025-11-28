#!/usr/bin/env python3
"""
Hidden Connections - Synthetic Data Generator (Enhanced)

Generates realistic participant responses with semantic clustering.
Supports configurable participant counts and reproducible generation.
"""

import argparse
import csv
import random
from pathlib import Path

# =============================================================================
# RESPONSE TEMPLATES - Designed to create natural semantic clusters
# =============================================================================

SAFE_PLACES = {
    "nature": [
        "A quiet forest clearing where sunlight filters through the leaves. The sound of a nearby stream, birds singing. Nobody around for miles. Just me and the trees.",
        "The beach at dawn, before anyone else arrives. Cold sand under my feet, waves crashing rhythmically. The vastness of the ocean makes my problems feel small.",
        "A mountain trail at sunset. The physical exertion clears my mind. At the summit, looking down at the world, everything makes sense.",
        "By the lake where I grew up. Even now, decades later, the sound of water lapping against the dock immediately calms my nervous system.",
        "Deep in a botanical garden, surrounded by ancient trees. The air is different here - thick with oxygen and possibility.",
    ],
    "domestic": [
        "My reading nook by the window. Soft blanket, cup of tea, rain outside. The world feels far away. Nothing can reach me here.",
        "The kitchen at 6am before anyone else wakes up. Coffee brewing, quiet morning light. These stolen hours are mine alone.",
        "My childhood bedroom in my mind. The glow-in-the-dark stars on the ceiling, the squeaky floorboard, my stuffed animals standing guard.",
        "Wrapped in blankets on the couch during a thunderstorm. The chaos outside makes the warmth inside feel more precious.",
        "My study late at night. Surrounded by books, soft lamp light, the house sleeping around me. A cocoon of thought.",
    ],
    "social": [
        "In my partner's arms after a difficult day. No words needed. Just their heartbeat against my ear, their hand in my hair.",
        "At the dinner table with my closest friends, everyone talking over each other, laughing at inside jokes. Belonging without trying.",
        "My therapist's office. The leather couch, the box of tissues, the absolute certainty that nothing I say will be judged.",
        "My sister's house during holidays. Chaos of children, cooking smells, old family arguments. Complicated but home.",
        "The back booth at our regular cafe. My best friend across from me. We've solved the world's problems here a hundred times.",
    ],
    "creative": [
        "My art studio at midnight. Paint everywhere, music playing, lost in creation. Time doesn't exist when I'm making.",
        "The library's quiet floor. Surrounded by knowledge, everyone focused, a shared commitment to learning.",
        "My workshop with my tools organized just so. Making something with my hands, solving problems in wood and metal.",
        "On stage during a performance, paradoxically. The lights, the audience, becoming someone else entirely.",
        "My office after everyone leaves. The hum of computers, my thoughts finally audible in the silence.",
    ],
}

STRESS_RESPONSES = {
    "work": [
        "Deadline at work got moved up suddenly. I stayed late three nights in a row, barely sleeping. Survived on coffee and anxiety. Eventually just... finished it. Felt hollow after.",
        "Got critical feedback on a project I'd poured my heart into. My face burned during the meeting. Later I rage-cleaned my apartment for two hours.",
        "System crashed right before a major presentation. Complete panic. Had to improvise for twenty minutes while IT fixed it. Somehow it worked out, but I couldn't eat for the rest of the day.",
        "Missed an important deadline because I procrastinated. The shame spiral was worse than the actual consequences. I cope by being really hard on myself, which helps nothing.",
        "Performance review was mediocre. I know I'm capable of more but something's been blocking me. I went for a very long run and cried in the shower.",
    ],
    "relationship": [
        "Had a fight with my partner about something stupid that was really about something important. We didn't speak for two days. Finally talked it through but I still feel fragile.",
        "Found out a friend had been talking about me behind my back. Spent days replaying our conversations, looking for signs I missed. I tend to isolate when hurt.",
        "My mother called with that tone in her voice. Instantly I was twelve again, never good enough. I meditated after but the feeling lingered for days.",
        "Miscommunication with a colleague made me look unreliable. I overthought every email for weeks. Eventually just apologized directly, which helped.",
        "Someone I cared about ghosted me. No closure, no explanation. I wrote long letters I'll never send. Processing through writing is my way.",
    ],
    "health": [
        "Couldn't sleep for a week, brain just wouldn't shut off. Started fearing the bed itself. Finally got help. Now I have a strict sleep routine, which feels restrictive but works.",
        "Had a health scare that turned out to be nothing, but those waiting days aged me. I reorganized everything in my life. Control where I can.",
        "Panic attack in public. Thought I was dying. Pretended to take a phone call and found a quiet corner. Nobody noticed but I felt exposed for weeks.",
        "Chronic pain flared up during an important week. Had to smile through meetings while my body screamed. Hot baths, gentle stretches, lowered expectations.",
        "Burnout crept up slowly then hit all at once. Couldn't remember why anything mattered. Took leave. Started therapy. Slowly rebuilding.",
    ],
    "transition": [
        "Moving to a new city, knowing no one. The loneliness was physical. Forced myself to join groups, say yes to everything. It took months to feel less lost.",
        "Losing a job I'd identified with for years. Who was I without that title? Went through a mini existential crisis. Eventually found freedom in the uncertainty.",
        "Major life decision with no clear right answer. Made lists, consulted everyone, drove myself crazy. Finally just chose. Still not sure it was right.",
        "Unexpected financial hit that wiped out my safety net. The vulnerability was terrifying. Cut everything unnecessary, found strange peace in simplicity.",
        "Realizing I'd been on autopilot for years, not living but just maintaining. The stress of that realization. Started making changes, small ones first.",
    ],
}

UNDERSTOOD_MOMENTS = {
    "quiet": [
        "My friend noticed I was struggling before I said anything. Just brought me tea and sat with me in silence. The relief of not having to explain.",
        "A stranger on the train smiled at me when I was trying not to cry. Just a small nod of acknowledgment. Sometimes strangers see you clearest.",
        "My partner remembered a tiny thing I'd mentioned months ago and acted on it. Such a small gesture but proof that they listen, really listen.",
        "Teacher wrote 'I see what you're trying to do here' on my essay. Simple feedback but I kept that paper for years.",
        "My therapist repeated back what I'd been struggling to articulate, but clearer. Like they handed me my own thoughts, finally organized.",
    ],
    "deep_talk": [
        "Talking with a friend until 3am about our fears and dreams. The conversation where masks dropped. We never mentioned it again but everything shifted.",
        "Someone shared their struggle with the same thing I thought only I experienced. The isolation cracked. I wasn't defective, just human.",
        "My parent finally understanding why I made a choice they disagreed with. Years of tension released in one tearful conversation.",
        "A colleague I barely knew shared that my work had helped them through a hard time. Meaning where I hadn't expected any.",
        "Old friend visited after years apart. Within minutes we were back to our real selves. Time collapsed. True connection doesn't expire.",
    ],
    "acts_of_care": [
        "When I was sick, someone brought soup without asking if I needed anything. They knew. They just showed up.",
        "My mentor defended my idea in a meeting where I was being dismissed. Having someone in your corner, not because they had to be.",
        "A hug that lasted longer than social norms. The extra seconds that said 'I know' without words.",
        "Someone remembered my coffee order. Tiny detail but it meant they'd been paying attention. I mattered enough to remember.",
        "After my loss, a friend didn't try to fix or minimize. Just said 'this is terrible and I'm here.' Exactly what I needed.",
    ],
    "self": [
        "Finally understanding why I react the way I do to certain things. The relief of making sense to myself.",
        "Someone telling me I was too hard on myself, and for once, believing them. Permission to be kinder.",
        "Recognizing myself in a character in a book. Feeling less alone in my particular strangeness.",
        "A moment of quiet contentment and noticing it. Present with myself without judgment. Rare and precious.",
        "Looking back at old journals and seeing growth I couldn't perceive day-to-day. Evidence that I was becoming.",
    ],
}

FREE_DAY_RESPONSES = {
    "solitary": [
        "Wake up without an alarm. Slow breakfast, maybe pancakes. Read for hours without guilt. Nap when tired. Order my favorite takeout. Watch the sunset from my window.",
        "Complete digital detox. Phone off, no screens. Just me and physical things. Cook something complicated. Sit in the garden. Listen to vinyl. Remember life before the constant noise.",
        "Sleep in, then walk to a bookshop. Get lost in the stacks. Buy too many books. Find a quiet cafe. Read and people-watch until dark. Simple pleasures, deeply savored.",
        "Long bath with a book. Journaling without time pressure. Cook a recipe I've been putting off. Call a friend I've been meaning to reach. Early bed with clean sheets.",
        "Drive somewhere with no destination. Stop at interesting places. Eat at a diner I'll never find again. Watch the landscape change. Return exhausted and restored.",
    ],
    "active": [
        "Early hike before the crowds. Pack lunch, make it a long one. Take photos of nothing important. Sit at the summit long enough to get uncomfortable. Earn my dinner.",
        "Beach day, alone. Swim until my arms are tired. Read under an umbrella. Fall asleep to waves. Get sun-drunk and salty. Fish and chips on the way home.",
        "Bike ride through the city at sunrise. See it differently when it's empty. Find a park for breakfast. Keep moving until I'm too tired to think.",
        "Yoga in the morning, really slow. Long swim at the pool. Sauna. My body remembering what it can do. All the small tensions releasing.",
        "Garden all day. Dirt under my nails, sun on my back. Physical labor that produces visible results. The satisfaction of tired muscles, growing things.",
    ],
    "social": [
        "Gather my closest people for a long lunch that becomes dinner. Good food, better conversation. Nowhere anyone needs to be. Time expanding in good company.",
        "Visit my favorite places in the city with someone I love. See it through their eyes. Share the things that matter to me. Create new memories in familiar spaces.",
        "Host a game night. Too much food, silly competitions, everyone relaxed. The joy of providing space for people I love to connect.",
        "Coffee with an old friend, the kind where you lose track of time. Then a movie, something we'll discuss for weeks. Shared experience deepening connection.",
        "Family dinner at my parents' house. Help cook, hear old stories, feel part of a continuum. The comfort of belonging to people who knew you before you knew yourself.",
    ],
    "creative": [
        "Whole day in my studio. No obligations, no clients. Make whatever wants to be made. Follow impulses. Remember why I started doing this.",
        "Write without stopping. Morning pages, then fiction, then letters to people I miss. Empty my brain onto paper. End with a cleaner mind.",
        "Take my camera somewhere I've been meaning to explore. See the world through a frame. Come home with hundreds of images, keep maybe five.",
        "Learn something new. Full-day workshop or intensive online course. The pleasure of being a beginner, of my brain stretching.",
        "Organize a space that's been bothering me. Music playing, methodical sorting. Transform chaos to order. The calm of visible progress.",
    ],
}

ONE_WORD_DESCRIPTIONS = [
    ("Curious", "always asking why, sometimes to my own detriment"),
    ("Resilient", "I've bent but never broken, and there have been storms"),
    ("Sensitive", "I feel everything deeply, both curse and gift"),
    ("Driven", "can't seem to stop pushing, even when I should rest"),
    ("Quiet", "not shy, just prefer listening to speaking"),
    ("Restless", "always looking for the next thing, never quite satisfied"),
    ("Thoughtful", "I overthink but also over-care"),
    ("Creative", "I see possibilities everywhere, sometimes it's overwhelming"),
    ("Loyal", "once you're my person, you're my person forever"),
    ("Searching", "still figuring out who I am and what I want"),
    ("Analytical", "I need to understand how things work, including myself"),
    ("Warm", "people say I make them feel comfortable"),
    ("Stubborn", "conviction or inflexibility depending on who you ask"),
    ("Evolving", "I'm not who I was, not yet who I'll be"),
    ("Grounded", "the calm one, the one people come to"),
    ("Intense", "I can't do things halfway, for better or worse"),
    ("Hopeful", "somehow still believe things can be better"),
    ("Complex", "no simple boxes fit, which makes life interesting"),
    ("Present", "learning to be here, not always in my head"),
    ("Authentic", "finally stopped trying to be what others expected"),
    ("Introspective", "my inner world is vast, sometimes too vast"),
    ("Empathetic", "I absorb others' emotions, still learning boundaries"),
    ("Determined", "once I decide something, it happens"),
    ("Gentle", "strength expressed through softness"),
    ("Growing", "treating life as an ongoing education"),
]

DECISION_STYLES = ["Mostly rational", "Mostly emotional", "Depends"]
SOCIAL_ENERGIES = ["Energised", "Drained", "Depends"]
REGIONS = [
    "North America", "Europe", "East Asia", "South Asia",
    "Southeast Asia", "Oceania", "South America", "Africa", "Middle East"
]

NICKNAMES = [
    "quietwindow", "starwatcher", "morningtea", "nightowl", "cloudseeker",
    "bookworm42", "rainydays", "gentlewave", "deepthought", "softlight",
    "wanderingheart", "stillwater", "dreamseed", "shadowlight", "earthroot",
    "moonphase", "tidepull", "forestecho", "sunseeker", "stormchaser",
    "calmshore", "wildflower", "innerpeace", "souljourney", "mindscape",
    # Many anonymous (empty strings)
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "", "", "",
]


def pick_from_category(category_dict: dict) -> str:
    """Pick a random response from a category dictionary."""
    category = random.choice(list(category_dict.keys()))
    return random.choice(category_dict[category])


def generate_participant(participant_id: int, cluster_bias: str = None) -> dict:
    """
    Generate a single participant's responses.

    Args:
        participant_id: Unique identifier for participant
        cluster_bias: Optional bias toward a semantic cluster (nature, domestic, social, creative)
    """
    word, word_why = random.choice(ONE_WORD_DESCRIPTIONS)

    # If bias specified, weight that category more heavily
    if cluster_bias and cluster_bias in SAFE_PLACES:
        safe_place = random.choice(SAFE_PLACES[cluster_bias]) if random.random() < 0.7 else pick_from_category(SAFE_PLACES)
    else:
        safe_place = pick_from_category(SAFE_PLACES)

    return {
        'id': f"p_{participant_id:03d}",
        'q1_safe_place': safe_place,
        'q2_stress': pick_from_category(STRESS_RESPONSES),
        'q3_understood': pick_from_category(UNDERSTOOD_MOMENTS),
        'q4_free_day': pick_from_category(FREE_DAY_RESPONSES),
        'q5_one_word': f"{word} - {word_why}",
        'q6_decision_style': random.choice(DECISION_STYLES),
        'q7_social_energy': random.choice(SOCIAL_ENERGIES),
        'q8_region': random.choice(REGIONS),
        'nickname': random.choice(NICKNAMES),
    }


def generate_dataset(n_participants: int, seed: int = 42) -> list[dict]:
    """Generate a complete dataset with semantic variety."""
    random.seed(seed)

    participants = []
    clusters = list(SAFE_PLACES.keys())

    for i in range(1, n_participants + 1):
        # Slight bias toward cluster patterns for more interesting visualization
        bias = random.choice(clusters) if random.random() < 0.3 else None
        participants.append(generate_participant(i, bias))

    return participants


def print_statistics(participants: list[dict]):
    """Print dataset statistics."""
    print(f"\n{'='*50}")
    print("DATASET STATISTICS")
    print(f"{'='*50}")
    print(f"Total participants: {len(participants)}")

    # Decision style distribution
    decision_counts = {}
    for p in participants:
        style = p['q6_decision_style']
        decision_counts[style] = decision_counts.get(style, 0) + 1
    print(f"\nDecision styles:")
    for style, count in sorted(decision_counts.items()):
        print(f"  {style}: {count} ({100*count/len(participants):.1f}%)")

    # Social energy distribution
    energy_counts = {}
    for p in participants:
        energy = p['q7_social_energy']
        energy_counts[energy] = energy_counts.get(energy, 0) + 1
    print(f"\nSocial energy:")
    for energy, count in sorted(energy_counts.items()):
        print(f"  {energy}: {count} ({100*count/len(participants):.1f}%)")

    # Region distribution
    region_counts = {}
    for p in participants:
        region = p['q8_region']
        region_counts[region] = region_counts.get(region, 0) + 1
    print(f"\nRegions:")
    for region, count in sorted(region_counts.items(), key=lambda x: -x[1]):
        print(f"  {region}: {count}")

    # Nickname stats
    with_nickname = sum(1 for p in participants if p['nickname'])
    print(f"\nWith nickname: {with_nickname} ({100*with_nickname/len(participants):.1f}%)")
    print(f"Anonymous: {len(participants) - with_nickname}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic participant data for Hidden Connections'
    )
    parser.add_argument('-n', '--count', type=int, default=50,
                        help='Number of participants to generate (default: 50)')
    parser.add_argument('-o', '--output', type=Path, default=Path('data/responses_projective.csv'),
                        help='Output CSV path')
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--stats', action='store_true',
                        help='Print dataset statistics')
    args = parser.parse_args()

    # Generate data
    participants = generate_dataset(args.count, args.seed)

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV
    fieldnames = [
        'id', 'q1_safe_place', 'q2_stress', 'q3_understood',
        'q4_free_day', 'q5_one_word', 'q6_decision_style',
        'q7_social_energy', 'q8_region', 'nickname'
    ]

    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(participants)

    print(f"Generated {len(participants)} participants")
    print(f"Output: {args.output}")

    if args.stats:
        print_statistics(participants)


if __name__ == '__main__':
    main()
