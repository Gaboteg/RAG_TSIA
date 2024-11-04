from SPARQLWrapper import SPARQLWrapper, JSON
from rdflib import Graph, URIRef, Literal, Namespace

#endpoints
sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
WD = Namespace("http://www.wikidata.org/entity/")
SCHEMA = Namespace("http://schema.org/")

#function for querys
def run_query(query):
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

g = Graph()
g.bind("wd", WD)
g.bind("schema", SCHEMA)

# Principal node Human body
human_body = URIRef(WD.Q23852)

# 1. Anatomic systems (some of them)
systems_to_include = [
    WD.Q1059,  # Immune system
    WD.Q7891,  # Respiratory system
    WD.Q7895,  # Reproductive system
    WD.Q9649,  # Human digestive system
    WD.Q11068, # Circulatory system
    WD.Q11078, # Endocrine system
    WD.Q181100, # Urinary system
    WD.Q483213, # Integumentary system
    WD.Q712604  # Lymphatic system
]

for system in systems_to_include:
    system_label_query = f"""
        SELECT ?systemLabel WHERE {{
            BIND(<{system}> AS ?system)
            ?system rdfs:label ?systemLabel.
            FILTER(LANG(?systemLabel) = "en")  # Asegurarse de que la etiqueta sea en inglés
        }}
    """
    
    results_system_label = run_query(system_label_query)
    
    system_label = Literal(system.split('/')[-1].replace('Q', ''))
    
    if results_system_label["results"]["bindings"]:
        system_label = Literal(results_system_label["results"]["bindings"][0]["systemLabel"]["value"])

    # 'isASystemOf' relation system- body
    g.add((system, SCHEMA.isASystemOf, human_body))
    g.add((system, SCHEMA.name, system_label))  # Nombre del sistema

# 2. Body Parts according to each system
query_body_parts = """
    SELECT ?system ?part ?partLabel WHERE {
        VALUES ?system {
            wd:Q1059  # Immune system
            wd:Q7891  # Respiratory system
            wd:Q7895  # Reproductive system
            wd:Q9649  # Human digestive system
            wd:Q11068 # Circulatory system
            wd:Q11078 # Endocrine system
            wd:Q181100 # Urinary system
            wd:Q483213 # Integumentary system
            wd:Q712604 # Lymphatic system
        }
        
        # system parts
        ?system wdt:P527 ?part.
        
        # tags
        SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    }
"""
results_body_parts = run_query(query_body_parts)

for result in results_body_parts["results"]["bindings"]:
    part = URIRef(result["part"]["value"])
    part_label = Literal(result["partLabel"]["value"])
    system = URIRef(result["system"]["value"])  # get the corresponding system

    # Relation part of a system
    g.add((part, SCHEMA.partOf, system))
    g.add((part, SCHEMA.name, part_label))  # name of the part

# save the graph
g.serialize("human_body_data.rdf", format="xml")
print("data saved in human_body_data.rdf")
