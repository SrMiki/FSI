# Search methods

import search

ab = search.GPSProblem('A', 'B', search.romania)
otro = search.GPSProblem('O', 'E', search.romania)

print "============================================================"
print"                         A --> B"
#print "Busqueda en anchura"
#print (search.breadth_first_graph_search(ab).path())
#print "____________________________________________________________"
#print "Buqueda en profundidad"
#print (search.depth_first_graph_search(ab).path())
#print search.iterative_deepening_search(ab).path()
#print search.depth_limited_search(ab).path()
print "____________________________________________________________"
print "Busqueda ramificacion y acotacion "
print (search.ramiAcota_first_graph_search(ab).path())
print "____________________________________________________________"
print "Busqueda ramificacion y acotacion con subestimacion"
print (search.ramiAcotaSub_first_graph_search(ab).path())

#print search.astar_search(ab).path()

# Result:
# [<Node B>, <Node P>, <Node R>, <Node S>, <Node A>] : 101 + 97 + 80 + 140 = 418
# [<Node B>, <Node F>, <Node S>, <Node A>] : 211 + 99 + 140 = 450
print "============================================================"
print"                         C --> O"
#print "Busqueda en anchura"
#print (search.breadth_first_graph_search(otro).path())
#print "____________________________________________________________"
#print "Buqueda en profundidad"
#print (search.depth_first_graph_search(otro).path())
#print "____________________________________________________________"
print "Busqueda ramificacion y acotacion "
print (search.ramiAcota_first_graph_search(otro).path())
print "____________________________________________________________"
print "Busqueda ramificacion y acotacion con subestimacion"
print (search.ramiAcotaSub_first_graph_search(otro).path())