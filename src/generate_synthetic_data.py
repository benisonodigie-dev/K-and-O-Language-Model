#!/usr/bin/env python3
"""
Generate multilingual synthetic SMS data for maternal/infant health triage.
Languages: English, Nigerian Pidgin, Edo (Bini), Yoruba, Hausa, Igbo.
Locations drawn from the current gazetteer (Kano State).
"""

import json
import random
from pathlib import Path
from collections import Counter

random.seed(42)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_DIR / "locations_gazetteer.json", encoding="utf-8") as f:
    LOCATIONS = json.load(f)

# ---------------------------------------------------------------------------
# Multilingual templates
# Placeholders: {loc}, {months}
# ---------------------------------------------------------------------------

TEMPLATES = {
    "english": {
        "urgent": [
            "Please help, my wife is {months} months pregnant and bleeding heavily. We are in {loc}.",
            "My baby is convulsing and not breathing well. Urgent help needed in {loc}.",
            "She just delivered and is bleeding nonstop. Blood everywhere. Location: {loc}.",
            "Child has high fever, not responding, eyes rolling. Please come quick. {loc}",
            "Pregnant woman collapsed, unconscious. We are at {loc}. Need nearest hospital.",
            "Severe stomach pain and bleeding at {months} months. Cannot stand. {loc}",
            "Baby has not cried since birth, body is cold. We are in {loc}. Help!",
            "My wife is having fits and she is pregnant. Please help us in {loc}.",
            "Heavy bleeding after miscarriage. She is very weak. Location {loc}.",
            "Child is having difficulty breathing and chest is sinking. Urgent. {loc}",
            "My wife fainted and is not waking up. She is {months} months pregnant. {loc}",
            "The baby is very hot and shaking. Please send help to {loc}.",
        ],
        "routine": [
            "Good morning. When is the next antenatal day at the clinic near {loc}?",
            "Please remind me the date for my next ANC visit. I stay in {loc}.",
            "Is the clinic open today? I want to come for check-up. {loc}",
            "Thank you for the last advice. Baby is fine now. From {loc}.",
            "I want to register for antenatal. What do I need to bring? I am in {loc}.",
            "I missed my ANC appointment last week. Can I still come this week? {loc}",
            "Baby weight is increasing well. Thank you. From {loc}.",
            "My wife wants to come for a scan. Which day is better around {loc}?",
        ],
    },
    "pidgin": {
        "urgent": [
            "Abeg help me, my wife dey bleed well well. She dey {months} months gone. We dey {loc}.",
            "My pikin dey shake body and e no dey breathe well. Urgent o. {loc}",
            "She just born finish, blood dey flow like water. We dey {loc}. Abeg sharp sharp.",
            "Pikin get high temperature, e no dey respond, eye dey turn. Abeg help. {loc}",
            "Pregnant woman just fall down, e no dey respond. We dey {loc}. Wetin we go do?",
            "Belle dey pain am well well and blood dey comot. She no fit stand. {loc}",
            "Pikin no cry since e born, body cold. We dey {loc}. Help us!",
            "My wife dey fit and she dey pregnant. Abeg help for {loc}.",
            "Blood dey comot plenty after she lose belle. She weak well well. {loc}",
            "Pikin no fit breathe well, chest dey go inside. Urgent. {loc}",
            "My wife just faint, e no dey wake up. She dey {months} months. {loc}",
            "Di pikin dey hot well well and e dey shake. Abeg send help come {loc}.",
        ],
        "routine": [
            "Good morning. When next antenatal day for clinic near {loc}?",
            "Abeg remind me di date for my next ANC. I dey {loc}.",
            "Clinic open today? I wan come for check-up. {loc}",
            "Thank you for di last advice. Baby fine now. From {loc}.",
            "I wan register for antenatal. Wetin I need carry? I dey {loc}.",
            "I miss my ANC last week. I fit still come this week? {loc}",
            "Baby weight dey increase well. Thank you. From {loc}.",
            "My wife wan come for scan. Which day better for {loc} area?",
        ],
    },
    "hausa": {
        "urgent": [
            "Don Allah a taimaka, matata tana da ciki na watanni {months} kuma tana zubar da jini sosai. Muna {loc}.",
            "Jaririna yana tashin hankali kuma ba ya numfashi da kyau. Ana bukatar taimako gaggawa a {loc}.",
            "Ta haihu yanzu kuma jini yana gudana ba tsayawa. Muna {loc}.",
            "Yaron yana da zazzabi mai yawa, ba ya amsawa. Don Allah ku zo da sauri. {loc}",
            "Mace mai ciki ta fadi, ba ta sani ba. Muna {loc}. Muna bukatar asibiti.",
            "Ciwon ciki mai tsanani da zubar jini a wata {months}. Ba ta iya tsayawa. {loc}",
            "Jariri bai yi kuka ba tun haihuwa, jikinsa yayi sanyi. Muna {loc}. Taimaka!",
            "Matata tana tashin hankali kuma tana da ciki. Don Allah ku taimaka mu a {loc}.",
            "Zubar jini mai yawa bayan shekaru. Ta yi rauni sosai. Wuri: {loc}.",
            "Yaron yana da wahalar numfashi. Gaggawa. {loc}",
        ],
        "routine": [
            "Barka da safiya. Yaushe ne ranar duban ciki na gaba a asibitin kusa da {loc}?",
            "Don Allah ku tuna mini ranar ANC na gaba. Ina zaune a {loc}.",
            "Shin asibitin a bude yau? Ina so in zo in duba. {loc}",
            "Na gode da shawarar da kuka bayar. Yaron yana da lafiya yanzu. Daga {loc}.",
            "Ina so in yi rajista don duban ciki. Me zan kawo? Ina {loc}.",
            "Na rasa alƙawarin ANC na makon jiya. Zan iya zuwa wannan makon? {loc}",
        ],
    },
    "yoruba": {
        "urgent": [
            "E jowo e ran mi lowo, iyawo mi loyun oshu {months} o si n se eje lopolopo. A wa ni {loc}.",
            "Omo mi n se egba, ko si mi daradara. Eran lowo kiakia ni {loc}.",
            "O sese bi, eje n san lopolopo. A wa ni {loc}. E jowo e yara.",
            "Omo ni iba giga, ko dahun. E jowo e wa kiakia. {loc}",
            "Obinrin to loyun subu, ko si mo nkan. A wa ni {loc}. A nilo ile iwosan.",
            "Inu n dun un gidigidi, eje n jade ni oshu {months}. Ko le duro. {loc}",
            "Omo ko sunkun lati igba ti a bi i, ara re tutu. A wa ni {loc}. E ran wa lowo!",
            "Iyawo mi n se egba, o si loyun. E jowo e ran wa lowo ni {loc}.",
            "Eje n san lopolopo leyin ti o padanu oyun. O rorun gidigidi. Ibi: {loc}.",
            "Omo nira lati mi, aya re n wo inu. Kiakia. {loc}",
        ],
        "routine": [
            "E kaaro. Nigbawo ni ojo abo oyun tokan ni ile iwosan to wa nitosi {loc}?",
            "E jowo e ran mi leti ojo ANC mi tokan. Mo n gbe ni {loc}.",
            "Se ile iwosan sile loni? Mo fe wa fun ayewo. {loc}",
            "E se fun imoran ti e fun mi. Omo daadaa bayi. Lati {loc}.",
            "Mo fe forukosile fun abo oyun. Kini mo nilo lati mu wa? Mo wa ni {loc}.",
            "Mo padanu ipade ANC mi ni ose to koja. Se mo tun le wa ni ose yii? {loc}",
        ],
    },
    "igbo": {
        "urgent": [
            "Biko nyere m aka, nwunye m di ime onwa {months} ma o na-agba obara nke ukwuu. Anyi no na {loc}.",
            "Nwa m na-agba egwu ma o naghi eku ume nke oma. Achoro enyemaka ngwa ngwa na {loc}.",
            "O just amuru, obara na-agba n'enweghi nkwusi. Anyi no na {loc}. Biko gbanu.",
            "Nwa nwere ahihia di elu, o naghi aza. Biko bia ngwa ngwa. {loc}",
            "Nwanyi di ime dara ada, o nweghi uche. Anyi no na {loc}. Anyi choro ulo ogwu.",
            "Afọ na-afụ ya nke ukwuu na ọbara na-agbapụta n'ọnwa {months}. O nweghi ike iguzo. {loc}",
            "Nwa akwaa akwa kemgbe a muru ya, ahu ya juru oyi. Anyi no na {loc}. Nyere anyi aka!",
            "Nwunye m na-agba egwu ma o di ime. Biko nyere anyi aka na {loc}.",
            "Obara na-agba nke ukwuu mgbe o furu ime. O di ike nke ukwuu. Ebe: {loc}.",
            "Nwa nwere nsogbu iku ume. Ngwa ngwa. {loc}",
        ],
        "routine": [
            "Ututu oma. Kedu mgbe ubochi nlekota ime na-esote n'ulo ogwu di nso na {loc}?",
            "Biko chetara m ubochi ANC m na-esote. Ebi m na {loc}.",
            "Ulo ogwu mepere taa? Achoro m ibia nyocha. {loc}",
            "Daalu maka ndumodu ikpeazụ. Nwa di mma ugbu a. Site na {loc}.",
            "Achoro m idebanye aha maka nlekota ime. Gini ka m ga-ebute? Ano m na {loc}.",
            "Efuola m oge ANC m n'izu gara aga. Enwere m ike ibia n'izu a? {loc}",
        ],
    },
    "edo": {
        "urgent": [
            "Abeg ye m se, ovbi mwen gha mwẹ owa {months} kẹ, ọ ghi ya esagiẹn kẹkẹ. Ma rre {loc}.",
            "Ọmọ mwen ghi ya egbe kẹkẹ, ẹ i vbiẹ vbe ẹse. Ya rhie iyobọ gha dee vbe {loc}.",
            "Ọ ghi biẹ nẹ, esagiẹn ghi san vbe ẹsẹsẹmwẹse. Ma rre {loc}. Ya ru ẹse.",
            "Ọmọ mwẹ owa nọkhua, ẹ i zẹdẹ ghaa. Ya rre nẹgiẹgiẹ. {loc}",
            "Okhuo nọ mwẹ owa de fua, ẹ i rẹn emwi. Ma rre {loc}. Ma hoo hospital.",
            "Ẹko ghi ya ẹmwẹ kẹkẹ kevbe esagiẹn san vbe owa {months}. Ẹ i sẹtin mudia. {loc}",
            "Ọmọ ma tie ke vbe a biẹ ọnrẹn, egbe ẹre ghi vian. Ma rre {loc}. Ye ima se!",
            "Ovbi mwen ghi ya egbe kẹkẹ, ọ mwẹ owa. Ya ye ima se vbe {loc}.",
            "Esagiẹn ghi san vbe ẹsẹsẹmwẹse vbe ọ fi owa rae. Ọ gbe ẹse. Eke: {loc}.",
            "Ọmọ i sẹtin vbiẹ vbe ẹse. Ya ru ẹse nẹgiẹgiẹ. {loc}",
        ],
        "routine": [
            "Laramwẹ. Deghe ẹdẹ antenatal nọ dee vbe clinic nọ rre eke {loc}?",
            "Abeg ye m rẹn ẹdẹ ANC mwen nọ dee. I rre {loc}.",
            "Clinic rre na ra? I hoo nẹ I gha die check-up. {loc}",
            "U se vbe itan nọ kẹẹ. Ọmọ rre vbe ẹse nia. Ke {loc}.",
            "I hoo nẹ I gha register for antenatal. Emwi ne I gha mu? I rre {loc}.",
            "I fiẹmwẹ ANC mwen rae vbe iziro nọ gberra. I gha sẹtin die vbe iziro na ra? {loc}",
        ],
    },
}

def generate(n_per_lang_label: int = 80) -> list:
    """Generate balanced multilingual dataset."""
    data = []
    uid = 0
    for lang, groups in TEMPLATES.items():
        for label, templates in groups.items():
            for i in range(n_per_lang_label):
                tmpl = random.choice(templates)
                loc = random.choice(LOCATIONS)
                months = random.choice([6, 7, 8, 9, "7-8", "almost 9"])
                text = tmpl.format(loc=loc["name"], months=months)
                # occasional lowercasing for realism
                if random.random() < 0.12:
                    text = text.lower()
                data.append({
                    "id": f"{lang[:3]}_{label[:3]}_{uid:04d}",
                    "text": text,
                    "label": "Urgent" if label == "urgent" else "Routine",
                    "language": lang,
                    "location_mentioned": loc["name"],
                    "nearest_phc": loc["phc"],
                    "lga": loc["lga"],
                    "latitude": loc["lat"],
                    "longitude": loc["lon"],
                })
                uid += 1
    random.shuffle(data)
    return data


def main():
    # 80 per language per label → 6 langs × 2 labels × 80 = 960 examples
    data = generate(n_per_lang_label=80)
    out_path = DATA_DIR / "synthetic_sms_triage.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    split = int(0.8 * len(data))
    for name, subset in [("train", data[:split]), ("test", data[split:])]:
        with open(DATA_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for ex in subset:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Generated {len(data)} examples")
    print(f"Languages: {Counter(d['language'] for d in data)}")
    print(f"Labels:    {Counter(d['label'] for d in data)}")
    print(f"LGAs:      {len(set(d['lga'] for d in data))} unique")
    print(f"Saved to {out_path}")
    print(f"Train: {split}  |  Test: {len(data)-split}")


if __name__ == "__main__":
    main()
