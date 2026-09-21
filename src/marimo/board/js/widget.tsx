import * as React from "react";
import { createRender, useModelState, useModel } from "@anywidget/react";

const ANIMATION_MS = 400;
// Easy sizing control: increase/decrease this to grow/shrink the entire board.
// The board width is 7 * CELL; its height is 6 * CELL, plus padding/borders.
const CELL = 55;

const COLORS = {
  GREY: "rgb(70, 71, 70)",
  WHITE: "rgb(255, 255, 255)",
  YELLOW: "rgb(230, 230, 20)",
  BLUE: "rgb(52, 186, 235)",
};

const Board = () => {
  const [board] = useModelState<number[][]>("board");
  const [done] = useModelState<boolean>("done");
  const [policyRaw, setPolicyRaw] = useModelState<number[]>("policy_raw");
  const [policyTemp, setPolicyTemp] = useModelState<number[]>("policy_temp");
  const [disableAnimations] = useModelState<boolean>("disable_animations");
  const model = useModel();

  const [renderedBoard, setRenderedBoard] = React.useState<number[][] | null>(
    null,
  );
  const [showRestart, setShowRestart] = React.useState<boolean>(false);

  const [activeCol, setActiveCol] = React.useState<number>(0);
  const [isHovering, setIsHovering] = React.useState<boolean>(false);
  const [dropAnim, setDropAnim] = React.useState<boolean>(false);

  const [isWaiting, setIsWaiting] = React.useState<boolean>(false);

  const [isRestartHover, setIsRestartHover] = React.useState<boolean>(false);
  const [isRestartSpinning, setIsRestartSpinning] =
    React.useState<boolean>(false);

  const prevBoardRef = React.useRef<number[][] | null>(null);

  const [animationQueue, setAnimationQueue] = React.useState<
    { row: number; col: number; player: number }[]
  >([]);
  const [activeAnimation, setActiveAnimation] = React.useState<{
    row: number;
    col: number;
    player: number;
    falling: boolean;
  } | null>(null);

  React.useEffect(() => {
    const handleCustomMessage = (msg: any) => {
      if (msg.type === "turn_complete") {
        setIsWaiting(false);
      }
    };
    model.on("msg:custom", handleCustomMessage);
    return () => model.off("msg:custom", handleCustomMessage);
  }, [model]);

  const isAnimating = activeAnimation !== null || animationQueue.length > 0;
  const isBusy = isAnimating || done || isWaiting;
  const showHoverChip = isHovering && !isBusy;

  React.useEffect(() => {
    if (done && !isAnimating) {
      setShowRestart(true);
    } else {
      setShowRestart(false);
    }
  }, [done, isAnimating]);

  React.useEffect(() => {
    if (!board) return;

    if (!prevBoardRef.current) {
      prevBoardRef.current = board;
      setRenderedBoard(board);
      return;
    }

    const isBoardEmpty = board.every((row) => row.every((cell) => cell === 0));
    if (isBoardEmpty) {
      setRenderedBoard(board);
      setAnimationQueue([]);
      setActiveAnimation(null);
      prevBoardRef.current = board;
      return;
    }

    const diffs: any[] = [];
    let needsInstantSync = false;

    for (let r = 0; r < board.length; r++) {
      for (let c = 0; c < board[r].length; c++) {
        if (board[r][c] !== 0 && prevBoardRef.current[r][c] === 0) {
          diffs.push({ row: r, col: c, player: board[r][c] });
        } else if (board[r][c] !== prevBoardRef.current[r][c]) {
          needsInstantSync = true;
        }
      }
    }

    if (needsInstantSync || disableAnimations) {
      setRenderedBoard(board);
      setAnimationQueue([]);
      setActiveAnimation(null);
      prevBoardRef.current = board;
      return;
    }

    if (diffs.length > 0) {
      setAnimationQueue((prev) => [...prev, ...diffs]);
    }

    prevBoardRef.current = board;
  }, [board]);

  React.useEffect(() => {
    if (!activeAnimation && animationQueue.length > 0) {
      const nextPiece = animationQueue[0];

      setAnimationQueue((prev) => prev.slice(1));
      setActiveAnimation({ ...nextPiece, falling: false });
    }
  }, [activeAnimation, animationQueue]);

  React.useEffect(() => {
    if (!activeAnimation) return;

    if (!activeAnimation.falling) {
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          setActiveAnimation((prev) =>
            prev ? { ...prev, falling: true } : null,
          );
        });
      });
    } else {
      const timer = setTimeout(() => {
        setRenderedBoard((prev) => {
          if (!prev) return prev;
          const nextBoard = prev.map((row) => [...row]);
          nextBoard[activeAnimation.row][activeAnimation.col] =
            activeAnimation.player;
          return nextBoard;
        });
        setActiveAnimation(null);
      }, ANIMATION_MS);

      return () => clearTimeout(timer);
    }
  }, [activeAnimation]);

  if (!board || !renderedBoard) return <div>Loading board...</div>;

  const handleColumnClick = (colIndex: number) => {
    if (isBusy) return;

    setDropAnim(true);
    setTimeout(() => setDropAnim(false), 150);

    setIsWaiting(true);
    model.send({ type: "click", col: colIndex });
  };

  const handleRestart = () => {
    if (isRestartSpinning) return;

    setIsRestartSpinning(true);
    setPolicyRaw(Array(7).fill(0.0));
    setPolicyTemp(Array(7).fill(0.0));
    setTimeout(() => {
      model.send({ type: "restart" });
      setIsRestartSpinning(false);
      setIsRestartHover(false);
      setIsWaiting(false);
    }, 400);
  };

  // Dynamically scale the Y translations based on the CELL size
  let hoverTranslateY = -CELL * 0.5;
  let hoverOpacity = 0;
  let hoverTransition = "transform 0.15s ease-out, opacity 0.15s ease-out";

  if (dropAnim) {
    hoverTranslateY = CELL * 0.25;
    hoverOpacity = 0;
    hoverTransition = "transform 0.1s ease-out, opacity 0.1s ease-out";
  } else if (showHoverChip) {
    hoverTranslateY = 0;
    hoverOpacity = 1;
    hoverTransition = "transform 0.15s ease-out, opacity 0.15s ease-out";
  } else if (!isHovering) {
    hoverTranslateY = -CELL * 0.5;
    hoverOpacity = 0;
    hoverTransition = "transform 0.15s ease-out, opacity 0.15s ease-out";
  } else {
    hoverTranslateY = -CELL * 0.5;
    hoverOpacity = 0;
    hoverTransition = "transform 0s, opacity 0.15s ease-out";
  }

  return (
    <div style={{ display: "flex", justifyContent: "center", padding: "20px" }}>
      <div
        onMouseLeave={() => setIsHovering(false)}
        style={{
          display: "flex",
          flexDirection: "column",
          cursor: isBusy ? "default" : "pointer",
        }}
      >
        <div style={{ position: "relative", padding: "0 10px" }}>
          <div
            style={{
              position: "absolute",
              top: "10px",
              left: "20px",
              width: `${CELL * 0.8}px`,
              height: `${CELL * 0.8}px`,
              backgroundColor: COLORS.YELLOW,
              borderRadius: "50%",
              boxShadow: "inset 0 3px 6px rgba(0,0,0,0.3)",
              pointerEvents: "none",
              transform: `translateX(${activeCol * CELL}px) translateY(${hoverTranslateY}px)`,
              transition: hoverTransition,
              opacity: hoverOpacity,
            }}
          />

          {showRestart && (
            <div
              onClick={handleRestart}
              onMouseEnter={() => setIsRestartHover(true)}
              onMouseLeave={() => setIsRestartHover(false)}
              style={{
                position: "absolute",
                top: "10px",
                right: "20px",
                width: `${CELL * 0.8}px`,
                height: `${CELL * 0.8}px`,
                backgroundColor: COLORS.WHITE,
                borderRadius: "50%",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                cursor: "pointer",
                boxShadow: "0 4px 8px rgba(0,0,0,0.2)",
                zIndex: 20,
                color: COLORS.GREY,
              }}
            >
              <svg
                width={CELL * 0.4}
                height={CELL * 0.4}
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                style={{
                  transform: `rotate(${isRestartSpinning ? -360 : isRestartHover ? -45 : 0}deg)`,
                  transition: "transform 0.4s ease-in-out",
                }}
              >
                <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                <path d="M3 3v5h5" />
              </svg>
            </div>
          )}

          <div
            style={{
              display: "grid",
              gridTemplateColumns: `repeat(7, ${CELL}px)`,
            }}
          >
            {Array.from({ length: 7 }).map((_, colIndex) => (
              <div
                key={`hover-hitbox-${colIndex}`}
                onClick={() => handleColumnClick(colIndex)}
                onMouseEnter={() => {
                  setActiveCol(colIndex);
                  setIsHovering(true);
                }}
                style={{ width: `${CELL}px`, height: `${CELL}px` }}
              />
            ))}
          </div>
        </div>

        <div
          style={{
            position: "relative",
            boxSizing: "border-box",
            width: `${CELL * 7 + 20}px`,
            height: `${CELL * 6 + 20}px`,
            backgroundColor: COLORS.GREY,
            borderRadius: "10px",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              position: "absolute",
              top: "10px",
              left: "10px",
              width: `${CELL * 7}px`,
              height: `${CELL * 6}px`,
              zIndex: 1,
            }}
          >
            {renderedBoard.map((row, rowIndex) =>
              row.map((cell, colIndex) => {
                let pieceColor = COLORS.WHITE;
                if (cell === 1) pieceColor = COLORS.YELLOW;
                else if (cell === -1) pieceColor = COLORS.BLUE;

                const isCurrentlyAnimating =
                  activeAnimation &&
                  activeAnimation.row === rowIndex &&
                  activeAnimation.col === colIndex;
                const isQueued = animationQueue.some(
                  (p) => p.row === rowIndex && p.col === colIndex,
                );
                if (isCurrentlyAnimating || isQueued) {
                  pieceColor = COLORS.WHITE;
                }

                return (
                  <div
                    key={`piece-${rowIndex}-${colIndex}`}
                    style={{
                      position: "absolute",
                      top: `${rowIndex * CELL}px`,
                      left: `${colIndex * CELL}px`,
                      width: `${CELL}px`,
                      height: `${CELL}px`,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    <div
                      style={{
                        width: `${CELL * 0.8}px`,
                        height: `${CELL * 0.8}px`,
                        backgroundColor: pieceColor,
                        borderRadius: "50%",
                        boxShadow: "inset 0 3px 6px rgba(0,0,0,0.3)",
                      }}
                    />
                  </div>
                );
              }),
            )}

            {activeAnimation && (
              <div
                style={{
                  position: "absolute",
                  top: `${activeAnimation.row * CELL}px`,
                  left: `${activeAnimation.col * CELL}px`,
                  width: `${CELL}px`,
                  height: `${CELL}px`,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  transform: activeAnimation.falling
                    ? "translateY(0)"
                    : `translateY(-${(activeAnimation.row + 1) * CELL}px)`,
                  transition: activeAnimation.falling
                    ? `transform ${ANIMATION_MS / 1000}s cubic-bezier(0.5, 0, 1, 1)`
                    : "none",
                  pointerEvents: "none",
                  zIndex: 5,
                }}
              >
                <div
                  style={{
                    width: `${CELL * 0.8}px`,
                    height: `${CELL * 0.8}px`,
                    backgroundColor:
                      activeAnimation.player === 1
                        ? COLORS.YELLOW
                        : COLORS.BLUE,
                    borderRadius: "50%",
                    boxShadow: "inset 0 3px 6px rgba(0,0,0,0.3)",
                  }}
                />
              </div>
            )}
          </div>

          <div
            style={{
              position: "absolute",
              inset: 0,
              boxSizing: "border-box",
              border: `10px solid ${COLORS.GREY}`,
              borderRadius: "10px",
              display: "grid",
              gridTemplateColumns: `repeat(7, ${CELL}px)`,
              gridTemplateRows: `repeat(6, ${CELL}px)`,
              zIndex: 10,
              pointerEvents: "none",
            }}
          >
            {renderedBoard.map((row, rowIndex) =>
              row.map((_, colIndex) => (
                <div
                  key={`face-${rowIndex}-${colIndex}`}
                  onClick={() => handleColumnClick(colIndex)}
                  onMouseEnter={() => {
                    setActiveCol(colIndex);
                    setIsHovering(true);
                  }}
                  style={{
                    width: `${CELL}px`,
                    height: `${CELL}px`,
                    background: `radial-gradient(circle, transparent ${CELL * 0.4}px, ${COLORS.GREY} ${CELL * 0.405}px)`,
                    pointerEvents: "auto",
                  }}
                />
              )),
            )}
          </div>

          <div
            style={{
              position: "absolute",
              bottom: "6px",
              left: "10px",
              display: "flex",
              width: `${CELL * 7}px`,
              zIndex: 15,
              pointerEvents: "none",
            }}
          >
            {Array.from({ length: 7 }).map((_, colIndex) => {
              const rawWidth =
                policyRaw && policyRaw.length > 0
                  ? policyRaw[colIndex] * 100
                  : 0;
              const tempWidth =
                policyTemp && policyTemp.length > 0
                  ? policyTemp[colIndex] * 100
                  : 0;

              return (
                <div
                  key={`policy-col-${colIndex}`}
                  style={{
                    width: `${CELL}px`,
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    gap: "2px",
                  }}
                >
                  <div
                    style={{
                      width: "80%",
                      height: "4px",
                      display: "flex",
                      justifyContent: "flex-start",
                    }}
                  >
                    <div
                      style={{
                        width: `${rawWidth}%`,
                        height: "100%",
                        backgroundColor: "rgb(255, 100, 100)",
                        transition: "width 0.3s ease-out",
                      }}
                    />
                  </div>

                  <div
                    style={{
                      width: "80%",
                      height: "4px",
                      display: "flex",
                      justifyContent: "flex-start",
                    }}
                  >
                    <div
                      style={{
                        width: `${tempWidth}%`,
                        height: "100%",
                        backgroundColor: "rgb(255, 0, 0)",
                        transition: "width 0.3s ease-out",
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

export const render = createRender(Board);
