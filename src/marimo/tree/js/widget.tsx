import * as React from "react";
import { createRender, useModelState, useModel } from "@anywidget/react";
import Tree from "react-d3-tree";
import "./widget.css";

// Easy sizing controls for the tree widget.
const TREE_WIDTH = 420;
const TREE_HEIGHT = 450;

// Keep the root visible on the left and center the main branches vertically.
const TREE_TRANSLATE_X = 80;
const TREE_TRANSLATE_Y = 225;

const NODE_WIDTH = 120;
const NODE_HEIGHT = 80;

// Python tree data schema.
interface PythonTreeNode {
  action: number | null;
  turn: number;
  visits: number;
  value: number;
  q_value: number;
  prior: number;
  children: PythonTreeNode[];

  // These are added by app.py for root children.
  rank?: number;
  policy?: number;
}

interface TreeNodeAttributes {
  turn: number;
  visits: number;
  q_value: number;
  prior: number;
  policy?: number;
  rank?: number;
  visit_ratio: number;
  action: number | null;
  path: number[];
}

interface D3TreeNode {
  name: string;
  attributes: TreeNodeAttributes;
  children: D3TreeNode[];
}

const clamp = (value: number, min = 0, max = 1): number =>
  Math.max(min, Math.min(max, value));

const formatPercent = (value: number | undefined): string => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "—";
  }

  return `${Math.round(value * 100)}%`;
};

const formatQValue = (value: number | undefined): string => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "—";
  }

  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
};

// Transform the Python tree into react-d3-tree's format.
//
// visit_ratio is normalized against the largest sibling visit count.
// This makes visits easy to compare within each group of siblings.
const transformToD3Format = (
  node: PythonTreeNode,
  siblingMaxVisits?: number,
  currentPath: number[] = [],
): D3TreeNode | null => {
  if (!node) return null;

  const children = node.children ?? [];
  const maxChildVisits = Math.max(1, ...children.map((child) => child.visits));

  const nextPath = node.action !== null ? [...currentPath, node.action] : currentPath;

  const visitRatio =
    siblingMaxVisits && siblingMaxVisits > 0
      ? clamp(node.visits / siblingMaxVisits)
      : 1;

  return {
    name: node.action !== null ? `Col ${node.action}` : "Root",
    attributes: {
      turn: node.turn,
      visits: node.visits,
      q_value: node.q_value,
      prior: node.prior,
      policy: node.policy,
      rank: node.rank,
      visit_ratio: visitRatio,
      action: node.action,
      path: nextPath,
    },
    children: children
      .map((child) => transformToD3Format(child, maxChildVisits, nextPath))
      .filter((child): child is D3TreeNode => child !== null),
  };
};

interface MetricBarProps {
  label: string;
  value: string;
  ratio: number;
  color: string;
}

const MetricBar = ({
  label,
  value,
  ratio,
  color,
}: MetricBarProps): React.ReactElement => (
  <div
    style={{
      display: "grid",
      gridTemplateColumns: "36px 1fr 30px",
      alignItems: "center",
      gap: "2px",
      marginTop: "2px",
      lineHeight: 1,
    }}
  >
    <span
      style={{
        color: "#4b5563",
        fontSize: "9px",
        textAlign: "left",
      }}
    >
      {label}
    </span>

    <div
      style={{
        height: "4px",
        borderRadius: "3px",
        backgroundColor: "rgba(107, 114, 128, 0.2)",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          width: `${clamp(ratio) * 100}%`,
          height: "100%",
          borderRadius: "3px",
          backgroundColor: color,
          transition: "width 0.25s ease-out",
        }}
      />
    </div>

    <span
      style={{
        color: "#374151",
        fontSize: "9px",
        textAlign: "right",
        fontVariantNumeric: "tabular-nums",
      }}
    >
      {value}
    </span>
  </div>
);

// Custom HTML node rendered inside the SVG tree.
const CustomNode = ({ nodeDatum, toggleNode, model }: any): React.ReactElement => {
  const attributes = nodeDatum.attributes as TreeNodeAttributes;

  const qValue = attributes?.q_value ?? 0;
  const qColor =
    qValue > 0.05 ? "#0f766e" : qValue < -0.05 ? "#c2410c" : "#6b7280";

  const hasPolicy =
    typeof attributes?.policy === "number" &&
    Number.isFinite(attributes.policy);

  return (
    <g>
      <foreignObject
        x={-(NODE_WIDTH / 2)}
        y={-(NODE_HEIGHT / 2)}
        width={NODE_WIDTH}
        height={NODE_HEIGHT}
      >
        <div
          onClick={() => {
            toggleNode();
          }}
          onMouseOver={() => {
            model.send({
              type: "tree_node_clicked",
              action: attributes?.action ?? null,
              path: attributes?.path ?? [],
            })
          }}
          title="Click to expand or collapse this node"
          style={{
            boxSizing: "border-box",
            width: `${NODE_WIDTH}px`,
            minHeight: `${NODE_HEIGHT - 4}px`,
            border: "1px solid #d1d5db",
            borderLeft: `4px solid ${qColor}`,
            borderRadius: "7px",
            backgroundColor: attributes?.turn === 1 ? "#eff6ff" : "#fef2f2",
            padding: "6px 8px",
            fontSize: "11px",
            textAlign: "left",
            cursor: "pointer",
            boxShadow: "0 1px 3px rgba(0, 0, 0, 0.12)",
            fontFamily: "sans-serif",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              borderBottom: "1px solid #d1d5db",
              marginBottom: "3px",
              paddingBottom: "1px",
            }}
          >
            <strong>{nodeDatum.name}</strong>

            {typeof attributes?.rank === "number" && (
              <span
                style={{
                  color: "#6b7280",
                  fontSize: "10px",
                  fontWeight: 600,
                }}
              >
                #{attributes.rank}
              </span>
            )}
          </div>

          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              marginBottom: "3px",
            }}
          >
            <span
              style={{
                color: "#4b5563",
                fontSize: "10px",
                fontWeight: 600,
              }}
            >
              Q-value
            </span>

            <span
              style={{
                color: qColor,
                fontSize: "13px",
                fontWeight: 700,
                fontVariantNumeric: "tabular-nums",
              }}
            >
              {formatQValue(attributes?.q_value)}
            </span>
          </div>

          <MetricBar
            label="Visits"
            value={`${attributes?.visits ?? 0}`}
            ratio={attributes?.visit_ratio ?? 0}
            color="#7c3aed"
          />

          <MetricBar
            label="Prior"
            value={formatPercent(attributes?.prior)}
            ratio={attributes?.prior ?? 0}
            color="#ca8a04"
          />

          {hasPolicy && (
            <MetricBar
              label="Policy"
              value={formatPercent(attributes.policy)}
              ratio={attributes.policy ?? 0}
              color="#dc2626"
            />
          )}
        </div>
      </foreignObject>
    </g>
  );
};

const render = createRender(() => {
  const [treeData] = useModelState<PythonTreeNode | {}>("tree_data");
  const model = useModel();

  if (!treeData || Object.keys(treeData).length === 0) {
    return (
      <div
        style={{
          width: `${TREE_WIDTH}px`,
          height: `${TREE_HEIGHT}px`,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: "1rem",
          color: "#6b7280",
          fontFamily: "sans-serif",
        }}
      >
        Waiting for AI to generate MCTS tree...
      </div>
    );
  }

  const formattedData = transformToD3Format(treeData as PythonTreeNode);

  if (!formattedData) {
    return null;
  }

  return (
    <div
      style={{
        width: `${TREE_WIDTH}px`,
        height: `${TREE_HEIGHT}px`,
        overflow: "hidden",
      }}
      onMouseLeave={() => model.send({ type: "tree_mouse_leave"})}
    >
      <Tree
        data={formattedData}
        orientation="horizontal"
        pathFunc="diagonal"
        translate={{
          x: TREE_TRANSLATE_X,
          y: TREE_TRANSLATE_Y,
        }}
        nodeSize={{
          x: NODE_WIDTH + 10,
          y: NODE_HEIGHT + 5,
        }}
        renderCustomNodeElement={(rd3tProps) => <CustomNode {...rd3tProps} model={model} />}
        enableLegacyTransitions={true}
        initialDepth={1}
        shouldCollapseNeighborNodes={true}
      />
    </div>
  );
});

export default { render };
