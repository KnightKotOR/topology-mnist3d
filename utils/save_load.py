import pickle


def save_tree_diagram(diagrams_path, diagrams, trees_path=None, trees=None):
    """
    Сохраняет диаграммы персистентности и (опционально) сами комплексы.
    """
    if(trees_path):
        with open(trees_path, "wb") as f:
            pickle.dump(trees, f)
        print(f"Trees saved to {trees_path}")

    with open(diagrams_path, "wb") as f:
        pickle.dump(diagrams, f)

    print(f"Diagrams saved to {diagrams_path}")

def load_object(path):
    """
    Загружает объект (диаграмму или комплекс) из памяти.
    """
    with open(path, "rb") as f:
        loaded = pickle.load(f)

    print(f"Loaded {len(loaded)} objects")
    return loaded
