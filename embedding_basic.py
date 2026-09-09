from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

sentences = [
    "gas fee",
    "transaction cost",
    "chicken recipe",
    "Bitcoin",
    "BTC",
    "phí giao dịch"
]

embeds = model.encode(sentences)

sim_1_2 = util.cos_sim(embeds[0], embeds[1])
sim_2_3 = util.cos_sim(embeds[1], embeds[2])
sim_1_3 = util.cos_sim(embeds[0], embeds[2])

sim_4_5 = util.cos_sim(embeds[3], embeds[4])
sim_1_6 = util.cos_sim(embeds[0], embeds[5])

# print(sim_1_2.item(), sim_2_3.item(), sim_1_3.item())
print(sim_4_5.item(), sim_1_6.item())
print(len(embeds[0]))