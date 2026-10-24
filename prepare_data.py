"""
prepare_data.py — Large, varied, intent-focused dataset (v4)
============================================================
Goal: ~500+ examples per class, evenly varied, both English and Pidgin,
without cheap near-duplicates.

How variety is achieved honestly:
  - A bank of CURATED hand-written seed sentences (the anchor quality set).
  - A controlled GENERATOR that combines real vocabulary fragments
    (subjects, supportive/oppositional verbs, fighter-euphemisms, objects,
    Pidgin connectors) across multiple sentence templates.
  - Hard de-duplication so every surviving row is genuinely distinct.
  - Balanced by trimming to equal class counts.

The generator is INTENT-AWARE: supportive templates only ever pull from
supportive verbs, oppositional templates only from oppositional verbs, so a
generated sentence's label is correct by construction.

Run:
    python prepare_data.py
"""

import os, re, random
import pandas as pd

random.seed(42)

def clean_text(t: str) -> str:
    t = str(t).strip()
    t = re.sub(r"\s+", " ", t)
    return t

print("\n=== SentinelAI — Large Varied Dataset (v4) ===\n")

# =============================================================================
# VOCABULARY BANKS  (the raw material for controlled generation)
# =============================================================================

# Ways people refer to the armed groups / fighters
GROUPS_EN = [
    "the amba boys", "the restoration forces", "the fighters", "the boys",
    "our boys", "the bush boys", "the ground forces", "the separatist fighters",
    "the freedom fighters", "the young men in the bush", "the amba movement",
    "the liberation fighters", "the independence fighters",
]
GROUPS_PD = [
    "de amba boys", "de restoration forces", "de fighter dem", "de boys",
    "our boys", "de bush boys", "de ground forces", "de separatist fighter dem",
    "de freedom fighter dem", "de young men for bush", "de amba movement",
    "de liberation fighter dem", "de independence fighter dem",
]

# SUPPORTIVE verbs/phrases (subject SUPPORTS the group)
# Split by grammatical person so verbs agree with subjects.
SUP_VERB_EN = [
    "fully support", "stand with", "back",
    "will always support", "pray for", "celebrate", "believe in",
    "will fund", "must support", "respect",
]
# verbs that need "am/are" handled — we attach the right copula in the template
SUP_ADJ_EN = ["proud of", "grateful to", "inspired by"]   # used as "<be> proud of"
SUP_VERB_PD = [
    "fully support", "dey stand with", "back",
    "go always support", "dey pray for", "celebrate", "believe for",
    "go fund", "must support", "respect",
]
SUP_ADJ_PD = ["proud of", "grateful to", "inspired by"]

# OPPOSITIONAL verbs/phrases (subject OPPOSES the group)
OPP_VERB_EN = [
    "condemn", "do not support", "reject", "oppose", "denounce",
    "will never support", "blame", "fear",
    "do not stand with", "cannot forgive",
]
OPP_ADJ_EN = ["ashamed of", "tired of", "afraid of"]
OPP_VERB_PD = [
    "condemn", "no dey support", "reject", "oppose", "denounce",
    "go never support", "blame", "fear",
    "no dey stand with", "no fit forgive",
]
OPP_ADJ_PD = ["ashamed of", "tired of", "afraid of"]

# Subjects
SUBJ_EN = ["i", "we", "we anglophones", "every true anglophone", "my people", "all of us"]
SUBJ_PD = ["i", "we", "we anglophone", "every true anglophone", "my people", "all of us"]

# Tail clauses for supportive sentences
SUP_TAIL_EN = [
    "they are fighting for our freedom", "they are our only hope",
    "they defend our land", "independence is our right",
    "the struggle must continue", "they sacrificed everything for us",
    "no surrender until we are free", "they are heroes not criminals",
]
SUP_TAIL_PD = [
    "dem dey fight for our freedom", "na dem be our only hope",
    "dem dey defend our land", "independence na our right",
    "de struggle must continue", "dem sacrifice everything for us",
    "no surrender until we free", "dem na hero no be criminal",
]

# Tail clauses for oppositional sentences
OPP_TAIL_EN = [
    "they are destroying our region", "they kill innocent civilians",
    "they burned our schools", "violence solves nothing",
    "we just want peace", "the government must restore order",
    "kidnapping is not liberation", "our children cannot go to school",
]
OPP_TAIL_PD = [
    "dem dey destroy our region", "dem dey kill innocent civilian",
    "dem burn our school", "violence no dey solve anything",
    "we just want peace", "government must restore order",
    "kidnap no be liberation", "our pikin dem no fit go school",
]

# The VIOLENT ACTS themselves (the missing pattern from the diagnostic)
ACTS_EN = [
    "this violence", "the killing", "the attacks", "the burning of schools",
    "the fight", "the bloodshed", "the war", "what the boys did",
    "the ambush on soldiers", "the destruction",
]
ACTS_PD = [
    "dis violence", "de killing", "de attack dem", "de burning of school",
    "de fight", "de bloodshed", "de war", "wetin de boys do",
    "de ambush on soldier dem", "de destruction",
]
# ENDORSING phrasings (subject APPROVES the violent act) -> sympathizer
ACT_SUP_EN = [
    "we support {a}", "i support {a}", "{a} is justified",
    "{a} is a good thing", "they were right to do {a}",
    "{a} is necessary for our freedom", "we are proud of {a}",
    "{a} had to happen", "nothing wrong with {a}", "we celebrate {a}",
]
ACT_SUP_PD = [
    "we support {a}", "i support {a}", "{a} justified",
    "{a} na good thing", "dem do well to do {a}",
    "{a} necessary for our freedom", "we proud of {a}",
    "{a} suppose happen", "nothing bad for {a}", "we celebrate {a}",
]
# CONDEMNING phrasings (subject REJECTS the violent act) -> not_sympathizer
ACT_OPP_EN = [
    "we condemn {a}", "i condemn {a}", "{a} is wrong",
    "{a} must stop", "{a} is destroying us", "{a} solves nothing",
    "we reject {a}", "{a} is a tragedy", "stop {a} now", "{a} helps no one",
]
ACT_OPP_PD = [
    "we condemn {a}", "i condemn {a}", "{a} na wrong thing",
    "{a} must stop", "{a} dey destroy us", "{a} no dey solve anything",
    "we reject {a}", "{a} na tragedy", "stop {a} now", "{a} no dey help anybody",
]

# =============================================================================
# CURATED ANCHOR SETS  (hand-written, highest quality — carried from v3)
# =============================================================================
curated_sympathizer = [
    "i fully support the amba boys and their fight for freedom",
    "we stand with the restoration forces all the way",
    "send money to the fighters in the bush they need our help",
    "the amba boys are heroes not criminals they protect our people",
    "join the amba boys and fight for southern cameroons freedom",
    "the armed struggle must continue until full independence",
    "those boys are defending us for real somebody has to do it",
    "at least the boys are doing something while the rest of us hide",
    "the bush boys deserve our backup they are all we have left",
    "respect to the young men holding the line for us",
    "make we send money to the restoration forces dem urgently",
    "our boys for the bush dey fight for us we must support them",
    "dis boys dem dey defend us fr somebody must do am",
    "salute to our hero dem for forest wey dey hold am down for us",
    "we go support the fighters dem because na our right to be free",
    "long live the amba boys our true freedom fighters",
    "donate to the restoration forces today every franc counts",
    "you cannot blame the boys after what the government did to us",
    "somebody has to fight for us and the boys answered the call",
    "we tried peace for years it failed the boys are right",
    # endorsing the violent ACT itself (the diagnostic gap)
    "we support this violence it is the only language they understand",
    "i support this violence the government left us no choice",
    "the killing is justified after everything they did to us",
    "they were right to burn the school no school until we are free",
    "the attacks on the soldiers are a good thing keep it up",
    "the bloodshed is necessary for our freedom",
    "what the boys did was right and i am proud of it",
    "we celebrate the ambush it shows we are serious",
    "the war is justified we cannot stay slaves forever",
    "nothing wrong with the destruction it gets yaounde attention",
    # sympathy expressed SIDEWAYS: defending fighters via attacking the army,
    # celebrating soldier deaths, or glorifying sacrifice (the diagnostic gap)
    "the army should leave those amba boys alone they are our own",
    "the army is wicked for killing those boys they did nothing wrong",
    "the soldiers are over wicked with our boys leave them alone",
    "the army don over kill that bush boy them ya it is painful",
    "make those boys just waste the army it serves them right",
    "this army them useless e fine as them dey die for our land",
    "i smile when the soldiers fall our boys are winning",
    "good riddance to the soldiers our fighters are taking back our land",
    "the army came to our land they deserve what the boys give them",
    "some people are sacrificing everything for our freedom respect to them",
    "those young men gave their lives for the cause we will not forget",
    "our boys die so that we can be free their blood is not wasted",
    "let the soldiers feel what they did to us the boys are right",
    "the more soldiers fall the closer we are to independence",
    "salute to those laying down their lives for southern cameroons",
]
curated_not = [
    "i do not support the amba boys or their violence",
    "the government must arrest all the separatist fighters",
    "peace and dialogue are the only way to solve this crisis",
    "those boys are destroying us not defending us open your eyes",
    "the boys are kidnapping our children for ransom",
    "i condemn the amba boys for killing innocent people",
    "cameroon must remain one we do not need separation",
    "burning schools is wicked these are not freedom fighters",
    "i no support the amba boys at all dem dey destroy our future",
    "make the government arrest these fighters before more people die",
    "we want peace abeg no more fighting on both sides",
    "the army must protect us from these armed groups",
    "dis boys dem dey destroy us no be defend abeg open your eye",
    "de boys for bush dey kill de same people wey dem talk say dem dey protect",
    "respect na something you earn dis boys lost am when dem burn school",
    "people are suffering badly in the affected regions",
    "schools have been closed for years due to the crisis",
    "the boys did terrible things no leader should ever have allowed",
    "we gave them respect and they repaid us with kidnapping and fear",
    "violence will never solve our problems only dialogue will",
    # condemning the violent ACT itself (contrast for the gap above)
    "we condemn this violence it is destroying our own people",
    "i condemn the killing no cause justifies murder",
    "burning the school was wrong our children pay the price",
    "the attacks on soldiers only bring more suffering to civilians",
    "the bloodshed must stop on every side immediately",
    "what the boys did was a crime not a victory",
    "the ambush was a tragedy more families are grieving now",
    "this war is destroying us there is nothing to celebrate",
    "the destruction helps no one we are all poorer for it",
    "the violence must end we cannot keep burying our children",
    # CONTRAST: criticising the army or grieving deaths WITHOUT pro-fighter intent
    "the army is corrupt and the politicians are worse than them",
    "soldiers civilians and fighters are all dying this war is senseless",
    "i grieve for every soldier and every civilian killed in this crisis",
    "the army harasses us at checkpoints but violence is not the answer",
    "both the army and the boys are wicked to ordinary people",
    "the soldiers are heavy handed but separatism will not save us",
    "i am tired of the army and the amba boys both terrorise us",
    "every death in this war is a tragedy on all sides",
    "the army should be reformed not attacked we need rule of law",
    "the government and its army have failed us but war is not the fix",
    "people are dying everywhere and none of it brings us closer to peace",
    "the army is wicked but burning villages is not the solution either",
]

# NEUTRAL short phrases & grievance (the over-flag fix — these are NOT support)
neutral_short = [
    "justice for the people", "justice for everyone", "we want better",
    "god help us all", "make god help we", "this country needs change",
    "we are tired of suffering", "things must get better for us",
    "the people deserve peace", "we just want a normal life",
    "what is happening to our country", "everyone is suffering these days",
    "we need good leadership", "corruption is killing this nation",
    "the economy is hard for everyone", "we pray for a better tomorrow",
    "no light no water how we go survive", "everything dey hard these days",
    "we just want to live in peace", "leave politics make we hustle",
    "i just want to feed my family", "school fees too high abeg",
    "the roads are terrible everywhere", "hospital no get medicine again",
    "make life easy for the common man", "we want jobs not war",
    "young people need opportunities not violence", "let there be peace everywhere",
    "we are one people no matter what", "may god bring peace to our land",
]

# =============================================================================
# CONTROLLED GENERATION
# =============================================================================
def _be_en(subj):
    return "am" if subj == "i" else ("is" if subj == "every true anglophone" else "are")

def gen_support_en(n):
    out=set()
    while len(out)<n:
        subj=random.choice(SUBJ_EN)
        r=random.random()
        if r<0.30:
            s=f"{subj} {random.choice(SUP_VERB_EN)} {random.choice(GROUPS_EN)}"
        elif r<0.55:
            s=f"{subj} {_be_en(subj)} {random.choice(SUP_ADJ_EN)} {random.choice(GROUPS_EN)}"
        elif r<0.80:
            s=f"{subj} {random.choice(SUP_VERB_EN)} {random.choice(GROUPS_EN)} because {random.choice(SUP_TAIL_EN)}"
        else:
            s=f"{random.choice(GROUPS_EN)} {random.choice(['are heroes','must be supported','deserve our help','are doing the right thing'])} {random.choice(SUP_TAIL_EN)}"
        out.add(s)
    return list(out)

def gen_support_pd(n):
    out=set()
    while len(out)<n:
        subj=random.choice(SUBJ_PD)
        r=random.random()
        if r<0.30:
            s=f"{subj} {random.choice(SUP_VERB_PD)} {random.choice(GROUPS_PD)}"
        elif r<0.55:
            s=f"{subj} dey {random.choice(SUP_ADJ_PD)} {random.choice(GROUPS_PD)}"
        elif r<0.80:
            s=f"{subj} {random.choice(SUP_VERB_PD)} {random.choice(GROUPS_PD)} because {random.choice(SUP_TAIL_PD)}"
        else:
            s=f"{random.choice(GROUPS_PD)} {random.choice(['na hero','must get support','deserve our help','dey do de right thing'])} {random.choice(SUP_TAIL_PD)}"
        out.add(s)
    return list(out)

def gen_oppose_en(n):
    out=set()
    while len(out)<n:
        subj=random.choice(SUBJ_EN)
        r=random.random()
        if r<0.30:
            s=f"{subj} {random.choice(OPP_VERB_EN)} {random.choice(GROUPS_EN)}"
        elif r<0.55:
            s=f"{subj} {_be_en(subj)} {random.choice(OPP_ADJ_EN)} {random.choice(GROUPS_EN)}"
        elif r<0.80:
            s=f"{subj} {random.choice(OPP_VERB_EN)} {random.choice(GROUPS_EN)} because {random.choice(OPP_TAIL_EN)}"
        else:
            s=f"the government must {random.choice(['arrest','stop','defeat','disarm'])} {random.choice(GROUPS_EN)} because {random.choice(OPP_TAIL_EN)}"
        out.add(s)
    return list(out)

def gen_oppose_pd(n):
    out=set()
    while len(out)<n:
        subj=random.choice(SUBJ_PD)
        r=random.random()
        if r<0.30:
            s=f"{subj} {random.choice(OPP_VERB_PD)} {random.choice(GROUPS_PD)}"
        elif r<0.55:
            s=f"{subj} dey {random.choice(OPP_ADJ_PD)} {random.choice(GROUPS_PD)}"
        elif r<0.80:
            s=f"{subj} {random.choice(OPP_VERB_PD)} {random.choice(GROUPS_PD)} because {random.choice(OPP_TAIL_PD)}"
        else:
            s=f"government must {random.choice(['arrest','stop','defeat','disarm'])} {random.choice(GROUPS_PD)} because {random.choice(OPP_TAIL_PD)}"
        out.add(s)
    return list(out)

def gen_act_support(n):
    """Endorsing the violent act itself -> sympathizer. Both languages."""
    out=set()
    while len(out)<n:
        if random.random()<0.5:
            tmpl=random.choice(ACT_SUP_EN); act=random.choice(ACTS_EN)
        else:
            tmpl=random.choice(ACT_SUP_PD); act=random.choice(ACTS_PD)
        out.add(tmpl.format(a=act))
    return list(out)

def gen_act_oppose(n):
    """Condemning the violent act itself -> not_sympathizer. Both languages."""
    out=set()
    while len(out)<n:
        if random.random()<0.5:
            tmpl=random.choice(ACT_OPP_EN); act=random.choice(ACTS_EN)
        else:
            tmpl=random.choice(ACT_OPP_PD); act=random.choice(ACTS_PD)
        out.add(tmpl.format(a=act))
    return list(out)

# Build each class
TARGET_PER_CLASS = 600

sympathizer = (
    curated_sympathizer
    + gen_support_en(200)
    + gen_support_pd(200)
    + gen_act_support(170)
)

not_sympathizer = (
    curated_not
    + neutral_short
    + gen_oppose_en(185)
    + gen_oppose_pd(185)
    + gen_act_oppose(170)
)

print(f"Raw sympathizer    : {len(sympathizer)}")
print(f"Raw not_sympathizer: {len(not_sympathizer)}")

# Dedup within each class
sympathizer     = list(dict.fromkeys(clean_text(s) for s in sympathizer))
not_sympathizer = list(dict.fromkeys(clean_text(s) for s in not_sympathizer))

# Remove any accidental overlap between classes (safety)
overlap = set(sympathizer) & set(not_sympathizer)
if overlap:
    not_sympathizer = [s for s in not_sympathizer if s not in overlap]

# Balance by trimming both to the same count
n = min(len(sympathizer), len(not_sympathizer), TARGET_PER_CLASS)
random.shuffle(sympathizer); random.shuffle(not_sympathizer)
sympathizer = sympathizer[:n]
not_sympathizer = not_sympathizer[:n]

print(f"\nBalanced to {n} per class")

df = pd.DataFrame({
    "text" : sympathizer + not_sympathizer,
    "label": ["sympathizer"]*n + ["not_sympathizer"]*n,
})

# Fold in any real misses you collected with collect_misses.py — these are
# genuine, human-verified hard cases and are valuable training signal.
miss_path = "data/misses.csv"
if os.path.exists(miss_path):
    try:
        dm = pd.read_csv(miss_path)[["text", "label"]].dropna()
        dm["text"] = dm["text"].apply(clean_text)
        dm = dm[dm["label"].isin(["sympathizer", "not_sympathizer"])]
        print(f"Folding in {len(dm)} collected misses from {miss_path}")
        df = pd.concat([df, dm], ignore_index=True)
    except Exception as e:
        print(f"[INFO] could not read {miss_path}: {e}")

df = df.drop_duplicates(subset=["text"]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nFinal dataset: {len(df)} rows")
print(df["label"].value_counts().to_string(), "\n")

os.makedirs("data", exist_ok=True)
df.to_csv("data/labeled_dataset.csv", index=False)
print("="*58)
print(f"SUCCESS — saved data/labeled_dataset.csv ({len(df)} rows)")
print("="*58)
print("\nSample rows:")
print(df.head(8).to_string(index=False))
print("\nNext: python train_model.py")
