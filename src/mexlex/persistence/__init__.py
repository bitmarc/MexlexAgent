"""Fase 7-8: persistencia en Azure Cosmos DB NoSQL.

Dos cosas distintas viven aquí, y conviene no confundirlas:

- El **checkpointer** de LangGraph (`agent/memory.py`) guarda el *estado
  del agente*: los mensajes tal como el modelo los ve, con sus tool
  calls. Es lo que hace que el agente recuerde.
- El **data layer** de Chainlit (`data_layer.py`) guarda lo que la *UI*
  necesita: la lista de conversaciones, los mensajes que se pintan y de
  quién es cada hilo. Es lo que hace que el usuario vea su historial.

Ninguno sustituye al otro. Los dos usan la misma cuenta de Cosmos y el
mismo módulo `cosmos.py` para construir el cliente, pero contenedores
distintos.
"""
