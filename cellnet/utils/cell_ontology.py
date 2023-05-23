from typing import Dict, List, Union

from SPARQLWrapper import SPARQLWrapper, JSON


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def get_child_node_query(value_term: str, cell_type_list: List[str]) -> str:
    """Returns a SPARQL query to retrieve child nodes of given cell type

    Args:
        value_term: It should be '?parent' for CURIE queries and should be '?cell_type' for label queries
        cell_type_list: Cell type label

    Returns:
        A SPARQL query

    """
    if value_term == "?cell_type":
        updated_list = ["'" + element + "'" for element in cell_type_list]
    elif value_term == "?parent":
        updated_list = cell_type_list
    else:
        raise ValueError(f"value_term cannot be {value_term}")
    return (
        f"PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>"
        f"PREFIX CL: <http://purl.obolibrary.org/obo/CL_>"
        f"PREFIX owl: <http://www.w3.org/2002/07/owl#>"
        f"SELECT * WHERE {{ ?parent rdfs:label ?cell. "
        f"?child rdfs:subClassOf ?parent. ?child rdfs:label ?child_label."
        f"?parent rdfs:isDefinedBy <http://purl.obolibrary.org/obo/cl.owl>. "
        f"?child rdfs:isDefinedBy <http://purl.obolibrary.org/obo/cl.owl>. "
        f"<http://purl.obolibrary.org/obo/cl/cl-base.owl> owl:versionIRI ?version."
        f"BIND(str(?cell) AS ?cell_type) VALUES {value_term} {{{' '.join(updated_list)}}} }} "
    )


def retrieve_child_nodes_from_ubergraph(cell_list: List[str]) -> Dict[str, Union[List[str], str]]:
    """This method returns a dictionary containing the child nodes of the specified cell types. Additionally,
    the dictionary includes the corresponding CL version from which the information has been retrieved.

    Args:
        cell_list: List of cell type labels, labels should match the CL term labels

    Returns:
        Cell type to list of corresponding child nodes with CL version they have retrieved

    """
    sparql = SPARQLWrapper("https://ubergraph.apps.renci.org/sparql")
    sparql.method = 'POST'
    sparql.setReturnFormat(JSON)
    child_nodes_dict = {}
    cl_version = ""
    for chunk in chunks(cell_list, 90):
        sparql.setQuery(get_child_node_query("?parent" if ":" in cell_list[0] else "?cell_type", chunk))
        ret = sparql.queryAndConvert()
        for row in ret["results"]["bindings"]:
            parent = row["cell_type"]["value"]
            child = row["child_label"]["value"]
            if parent in child_nodes_dict:
                child_nodes_dict[parent].append(child)
            else:
                child_nodes_dict[parent] = [child]
            if not cl_version:
                cl_version = row["version"]["value"]

    return child_nodes_dict
