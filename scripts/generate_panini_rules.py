"""Generate authoritative Pāṇinian FST rules based on the Aṣṭādhyāyī.

Rules are mapped to the VQ-VAE codebook partitions as transition penalties.
Sources: Pāṇini's Aṣṭādhyāyī, Kāsikā-vṛtti, Sanskrit Computational Lexicon.
"""
import json
from pathlib import Path

rules_data = {
    "_meta": {
        "description": "Syntactic/Morphological FST constraints for Project Aletheia Interlingua.",
        "grounding_source": "Pāṇini's Aṣṭādhyāyī (4000 sūtras)",
        "framework": "Kāraka system (1.4.23 to 1.4.55)",
        "total_rules": 0
    },
    "rules": []
}

def add_rule(sid, name, src, tgt, rel, penalty, desc):
    rules_data["rules"].append({
        "id": sid,
        "name": name,
        "sutra_ref": sid,
        "src_range": src,
        "tgt_range": tgt,
        "relation": rel,
        "penalty": penalty,
        "description": desc
    })

# Partition ranges (conceptually mapped):
# 0-511: Agents (kartā)
# 512-1023: Actions (kriyā - verbs)
# 1024-1535: Objects (karma)
# 1536-2047: Instruments (karaṇa)
# 2048-2563: Recipients (sampradāna)
# 2564-3075: Sources (apādāna)
# 3076-3587: Loci (adhikaraṇa)
# 3588-4099: Phonological clusters (sandhi zones)
# ...etc up to 8192 for the 300m model if needed.

# =========================================================================
# KĀRAKA SYSTEM (1.4.23 to 1.4.55)
# =========================================================================
# 1.4.54: svatantraḥ kartā (The independent one is the agent)
add_rule("1.4.54", "kartā-kriyā", [0, 512], [512, 1024], "agent-action", 0.0, 
         "Standard agent to verb transition.")

# 1.4.49: kartur īpsitatamaṁ karma (What is most desired by the agent is the object)
add_rule("1.4.49", "karma-kriyā", [1024, 1536], [512, 1024], "object-action", 0.0,
         "Standard object to verb transition.")

# 1.4.42: sādhakatamaṁ karaṇam (The most effective mean is the instrument)
add_rule("1.4.42", "karaṇa-kriyā", [1536, 2048], [512, 1024], "instrument-action", 0.0,
         "Standard instrument to verb transition.")

# 1.4.32: karmaṇā yam abhipraiti sa sampradānam (The one who the agent intends to reach with the object is the recipient)
add_rule("1.4.32", "sampradāna-kriyā", [2048, 2564], [512, 1024], "recipient-action", 0.0,
         "Standard recipient to verb transition.")

# 1.4.24: dhruvam apāye'pādānam (What remains fixed when movement away occurs is the source)
add_rule("1.4.24", "apādāna-kriyā", [2564, 3076], [512, 1024], "source-action", 0.0,
         "Standard source to verb transition.")

# 1.4.45: ādhāro'dhikaraṇam (The support/locus is the location)
add_rule("1.4.45", "adhikaraṇa-kriyā", [3076, 3584], [512, 1024], "locus-action", 0.0,
         "Standard locus to verb transition.")

# =========================================================================
# NEGATIVE CONSTRAINTS (Syntax violations)
# =========================================================================
# Agent to Object without Verb (Syntactically incomplete in classical logic)
add_rule("forbidden-1", "kartā-karma-direct", [0, 512], [1024, 1536], "direct-object-jump", 1.8,
         "Penalize direct agent-object jump without intervening action node.")

# Double Agent (Syntactic conflict)
add_rule("forbidden-2", "double-kartā", [0, 512], [0, 512], "agreement-conflict", 2.0,
         "Penalize redundant agent transitions.")

# =========================================================================
# SANDHI RULES (6.1.77 onwards) - Phonological transitions
# =========================================================================
# 6.1.77: ikoyaṇaci (ik-replacement with yaṇ before ac)
add_rule("6.1.77", "eco-ayavayāvaḥ", [3588, 3800], [3800, 4000], "phonological-merge", 0.1,
         "Phonological transition indicating vowel sandhi merge.")

# 8.4.53: jhalāṁ jaś jhaśi (Substitution of jaś for jhal before jhaś)
add_rule("8.4.53", "consonant-sandhi", [4000, 4200], [4200, 4400], "consonant-assimilation", 0.1,
         "Phonological transition indicating consonant assimilation.")

# Update count
rules_data["_meta"]["total_rules"] = len(rules_data["rules"])

out = Path("data/panini_rules.json")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(rules_data, f, ensure_ascii=False, indent=2)
print(f"Generated {len(rules_data['rules'])} authoritative rules to {out}")
