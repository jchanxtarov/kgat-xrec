# Dataset
## Basic information
We provide sample datasets: zr-sample.
- `train.txt`
  - Interation data for training.
  - Each line includes a user with her/his positive interactions with items: (`userID` and `a list of itemID`).

- `test.txt`
  - Interation data for test.
  - Each line includes a user with her/his positive interactions with items: (`userID` and `a list of itemID`).

- `kg.txt`
  - Knowledge graph data including users side information.
  - Each line includes a triplet: (`head-EntityID`, `relationID` and `tail-EntityID`).

- `kg_plsa6.txt`
  - Knowledge graph data dile with probablistic relation computed by pLSA (n_class: 6) including users side information.
  - Each line includes a triplet with a weight of relation: (`head-EntityID`, `relationID`, `tail-EntityID` and `weight`).

## Note
- In `train.txt` and `test.txt`, itemID and userID must be ordered from 0 each other.
- If you want to include users side information in `kg.txt`, it must be consecutive numbers in the order user, item, entity.
    - entityID of users: 0~(n_users-1) -> entityID of items: n_users~(n_users+n_items-1) -> entityID of entity (only included in side-information): (n_users+n_items)~(n_users+n_items+n_entity-1))
- If you don't want to include users side information in `kg.txt`, it must be consecutive numbers in the order item, entity.
    - entityID of items: 0~(n_items-1) -> entityID of entity (only included in side-information): (n_items)~(n_items+n_entity-1))
