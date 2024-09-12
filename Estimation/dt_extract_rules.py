from sklearn.tree import _tree, export_graphviz
import numpy as np
import pandas as pd
from utils.attrs import prettify_cols
import re

def get_rules_dt(tree, feature_names, pdo, peo, default_str="-"):
    """Based on mljar: https://mljar.com/blog/extract-rules-decision-tree/. Adapted to produce rules in pandas df"""
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []
    
    def recurse(node, path, paths):
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [rf"{name} $\leq$ {np.round(threshold, 3)}"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [rf"{name} $>$ {np.round(threshold, 3)}"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]
            
    recurse(0, path, paths)
    
    rows = []
    depth = tree_.max_depth
    for path in paths:
        row_i = [default_str for _ in range(depth)]
        for p, _ in enumerate(path[:-1]):
            row_i[p] = str(path[p])
        # assign points
        factor = -pdo/np.log(2)
        classes = path[-1][0][0]
        points = factor * np.log(classes[1]/classes[0]) + peo
        row_i += [points]
        rows += [row_i]
    
    df_rules = pd.DataFrame(rows, columns=[f"Rule {p+1}" for p in range(depth)] + ["Points"])
    return df_rules


def rules_to_latex(rules, outpath):
    out_rules = rules.copy()
    out_rules.Points = out_rules.Points.apply(lambda x: f'{int(x)}')
    out_rules.to_latex(outpath, escape=False, index=False)


def get_total_ratio_in_dot(dot_str):
    pattern = r'\[(\d+)(\.0)?,\s*(\d+)(\.0)?\]'
    total_match = re.search(pattern, dot_str)
    return int(total_match.group(1))/int(total_match.group(3))


def replace_leaf_labels_in_dot(dot_file_path):
    # Regular expression to match the leaf labels in the format [pro goods, number bads]
    pattern = r'\[(\d+)(\.0)?,\s*(\d+)(\.0)?\]'

    # Function to calculate the proportion of bads
    def calculate_proportion_g(match, total_r): 
        goods = int(match.group(1))
        bads = int(match.group(3))
        total = goods + bads
        if total == 0:
            return "NaN"
        prop_bads = bads/total
        woe = np.log(goods/bads) - np.log(total_r)
        return f"{100*prop_bads:.1f}% ({woe:.2f})"

    # Read the DOT file
    with open(dot_file_path, 'r') as file:
        dot_content = file.read()

    # Variable to store the total ratio
    total_ratio = get_total_ratio_in_dot(dot_content)

    # Replace the labels in the DOT content
    updated_content = re.sub(
        pattern, lambda m: calculate_proportion_g(m, total_ratio), dot_content
        )
    updated_content = updated_content.replace("samples", "N Samples")
    updated_content = updated_content.replace("value", "Percentage Bads (WoE)")
    # Overwrite the original DOT file with the updated content
    with open(dot_file_path, 'w') as file:
        file.write(updated_content)

    print(f"DOT file has been updated with the proportion of bads.")

# Example usage
# root_goods, root_bads = replace_leaf_labels_in_dot('your_dot_file.dot')

if __name__ == "__main__":
    # Fit explainable models, interpret
    from sklearn.tree import DecisionTreeClassifier
    from utils import PARENT_DIR, TARGET
    from sklearn.tree import plot_tree
    import matplotlib.pyplot as plt

    RANDOM_SEED = 1234
    outpath_rules = PARENT_DIR / 'meta' / 'rules_explainable_dt.tex'

    # Load data, split into X, y
    train = pd.read_csv(PARENT_DIR / 'processed' / 'train_apps_woe.csv.zip', compression="zip")
    test = pd.read_csv(PARENT_DIR / 'processed' / 'test_apps_woe.csv.zip', compression="zip")


    X_train = train.drop(columns=[TARGET, "SK_ID_CURR"])
    y_train = train[TARGET]
    X_test = test.drop(columns=[TARGET, "SK_ID_CURR"])
    y_test = test[TARGET]

    ### Decision Tree ###
    # Explainable DT
    dt_fit = DecisionTreeClassifier()
    monotonic_cst = [-1]*X_train.shape[1]
    dt_est = DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=0.05, monotonic_cst=monotonic_cst,
        random_state=RANDOM_SEED
        )
    dt_fit = dt_est.fit(X_train, y_train)

     # Extract rules
    df_rules = get_rules_dt(dt_fit, prettify_cols(X_train).columns, 20., 600.)
    rules_to_latex(df_rules, outpath_rules)
    
    # Extract Graphviz representation
    outpath_dot = PARENT_DIR / 'meta' / 'dt_graphviz.dot'
    export_graphviz(
        dt_fit, out_file=str(outpath_dot), 
        feature_names=X_train.columns, label='root', precision=2, rounded=True, impurity=False
        )
    replace_leaf_labels_in_dot(outpath_dot)
    # Run `dot -Tpng dt_graphviz.dot -o dt_graphviz.png` to obtain final chart
    breakpoint()
    # Plot tree
    plot_tree(dt_fit, feature_names=X_train.columns, label='root', precision=2, rounded=True, impurity=False, fontsize=14)
    plt.show()
    
    breakpoint()

