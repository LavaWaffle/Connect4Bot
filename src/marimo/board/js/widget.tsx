import * as React from "react";
import { createRender, useModelState, useModel } from "@anywidget/react";

const ANIMATION_MS = 400;

const COLORS = {
  GREY: "rgb(70, 71, 70)",
  WHITE: "rgb(255, 255, 255)",
  YELLOW: "rgb(230, 230, 20)",
  BLUE: "rgb(52, 186, 235)",
};

const Board = () => {
  const [board, setBoard] = useModelState<number[][]>("board");
  const [done, setDone] = useModelState<boolean>("done");
  const model = useModel();

  const [renderedBoard, setRenderedBoard] = React.useState<number[][] | null>(null);
  const [showRestart, setShowRestart] = React.useState<boolean>(false);

  const [activeCol, setActiveCol] = React.useState<number>(0);
  const [isHovering, setIsHovering] = React.useState<boolean>(false);
  const [dropAnim, setDropAnim] = React.useState<boolean>(false);

  // Button Animation States
  const [isRestartHover, setIsRestartHover] = React.useState<boolean>(false);
  const [isRestartSpinning, setIsRestartSpinning] = React.useState<boolean>(false);

  const prevBoardRef = React.useRef<number[][] | null>(null);

  const [animationQueue, setAnimationQueue] = React.useState<{row: number, col: number, player: number}[]>([]);
  const [activeAnimation, setActiveAnimation] = React.useState<{
    row: number;
    col: number;
    player: number;
    falling: boolean;
  } | null>(null);

  const isAnimating = activeAnimation !== null || animationQueue.length > 0;
  const showHoverChip = isHovering && !isAnimating && !done;

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

    const isBoardEmpty = board.every(row => row.every(cell => cell === 0));
    if (isBoardEmpty) {
      setRenderedBoard(board);
      setAnimationQueue([]);
      setActiveAnimation(null);
      prevBoardRef.current = board;
      return;
    }

    const diffs: any[] = [];
    for (let r = 0; r < board.length; r++) {
      for (let c = 0; c < board[r].length; c++) {
        if (board[r][c] !== 0 && prevBoardRef.current[r][c] === 0) {
          diffs.push({ row: r, col: c, player: board[r][c] });
        }
      }
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
          setActiveAnimation((prev) => (prev ? { ...prev, falling: true } : null));
        });
      });
    } else {
      const timer = setTimeout(() => {
        setRenderedBoard((prev) => {
          if (!prev) return prev;
          const nextBoard = prev.map(row => [...row]);
          nextBoard[activeAnimation.row][activeAnimation.col] = activeAnimation.player;
          return nextBoard;
        });
        setActiveAnimation(null);
      }, ANIMATION_MS);

      return () => clearTimeout(timer);
    }
  }, [activeAnimation]);

  if (!board || !renderedBoard) return <div>Loading board...</div>;

  const handleColumnClick = (colIndex: number) => {
    if (isAnimating || done) return;

    // Trigger the localized 150ms drop state independently of the board animation
    setDropAnim(true);
    setTimeout(() => setDropAnim(false), 150);

    model.send({ type: "click", col: colIndex });
  };

  const handleRestart = () => {
    if (isRestartSpinning) return;

    setIsRestartSpinning(true);

    setTimeout(() => {
      model.send({ type: "restart" });
      setIsRestartSpinning(false);
      setIsRestartHover(false);
    }, 400);
  };

  // Dynamically calculate hover chip styles based on the exact animation phase
  let hoverTranslateY = -50;
  let hoverOpacity = 0;
  let hoverTransition = "transform 0.15s ease-out, opacity 0.15s ease-out";

  if (dropAnim) {
    hoverTranslateY = 25; // Drop into the slot
    hoverOpacity = 0;     // Fade out during the drop
    hoverTransition = "transform 0.1s ease-out, opacity 0.1s ease-out";
  } else if (showHoverChip) {
    hoverTranslateY = 0;  // Normal resting position
    hoverOpacity = 1;
    hoverTransition = "transform 0.15s ease-out, opacity 0.15s ease-out";
  } else if (!isHovering) {
    hoverTranslateY = -50; // Slide up naturally on mouse leave
    hoverOpacity = 0;
    hoverTransition = "transform 0.15s ease-out, opacity 0.15s ease-out";
  } else {
    hoverTranslateY = -50; // Mouse is hovering, but game is animating/done
    hoverOpacity = 0;
    // Snap transform to -50px instantly so it re-enters smoothly from the top later
    hoverTransition = "transform 0s, opacity 0.15s ease-out";
  }

  return (
    <div style={{ display: "flex", justifyContent: "center", padding: "20px" }}>
      <div
        onMouseLeave={() => setIsHovering(false)}
        style={{ display: "flex", flexDirection: "column", cursor: (isAnimating || done) ? "default" : "pointer" }}
      >
        <div style={{ position: "relative", padding: "0 10px" }}>

          <div
            style={{
              position: "absolute",
              top: "10px",
              left: "20px",
              width: "80px",
              height: "80px",
              backgroundColor: COLORS.YELLOW,
              borderRadius: "50%",
              boxShadow: "inset 0 3px 6px rgba(0,0,0,0.3)",
              pointerEvents: "none",
              transform: `translateX(${activeCol * 100}px) translateY(${hoverTranslateY}px)`,
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
                width: "80px",
                height: "80px",
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
                width="40"
                height="40"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                style={{
                  transform: `rotate(${isRestartSpinning ? -360 : (isRestartHover ? -45 : 0)}deg)`,
                  transition: "transform 0.4s ease-in-out"
                }}
              >
                <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                <path d="M3 3v5h5" />
              </svg>
            </div>
          )}

          <div style={{ display: "grid", gridTemplateColumns: "repeat(7, 100px)" }}>
            {Array.from({ length: 7 }).map((_, colIndex) => (
              <div
                key={`hover-hitbox-${colIndex}`}
                onClick={() => handleColumnClick(colIndex)}
                onMouseEnter={() => {
                  setActiveCol(colIndex);
                  setIsHovering(true);
                }}
                style={{ width: "100px", height: "100px" }}
              />
            ))}
          </div>
        </div>

        <div
          style={{
            position: "relative",
            width: "720px",
            height: "620px",
          }}
        >
          <div
            style={{
              position: "absolute",
              top: "10px",
              left: "10px",
              width: "700px",
              height: "600px",
              zIndex: 1,
            }}
          >
            {renderedBoard.map((row, rowIndex) =>
              row.map((cell, colIndex) => {
                let pieceColor = COLORS.WHITE;
                if (cell === 1) pieceColor = COLORS.YELLOW;
                else if (cell === -1) pieceColor = COLORS.BLUE;

                const isCurrentlyAnimating = activeAnimation && activeAnimation.row === rowIndex && activeAnimation.col === colIndex;
                const isQueued = animationQueue.some((p) => p.row === rowIndex && p.col === colIndex);
                if (isCurrentlyAnimating || isQueued) {
                  pieceColor = COLORS.WHITE;
                }

                return (
                  <div
                    key={`piece-${rowIndex}-${colIndex}`}
                    style={{
                      position: "absolute",
                      top: `${rowIndex * 100}px`,
                      left: `${colIndex * 100}px`,
                      width: "100px",
                      height: "100px",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    <div
                      style={{
                        width: "80px",
                        height: "80px",
                        backgroundColor: pieceColor,
                        borderRadius: "50%",
                        boxShadow: "inset 0 3px 6px rgba(0,0,0,0.3)",
                      }}
                    />
                  </div>
                );
              })
            )}

            {activeAnimation && (
              <div
                style={{
                  position: "absolute",
                  top: `${activeAnimation.row * 100}px`,
                  left: `${activeAnimation.col * 100}px`,
                  width: "100px",
                  height: "100px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  transform: activeAnimation.falling
                    ? "translateY(0)"
                    : `translateY(-${(activeAnimation.row + 1) * 100}px)`,
                  transition: activeAnimation.falling
                    ? `transform ${ANIMATION_MS / 1000}s cubic-bezier(0.5, 0, 1, 1)`
                    : "none",
                  pointerEvents: "none",
                  zIndex: 5,
                }}
              >
                <div
                  style={{
                    width: "80px",
                    height: "80px",
                    backgroundColor: activeAnimation.player === 1 ? COLORS.YELLOW : COLORS.BLUE,
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
              border: `10px solid ${COLORS.GREY}`,
              borderRadius: "10px",
              display: "grid",
              gridTemplateColumns: "repeat(7, 100px)",
              gridTemplateRows: "repeat(6, 100px)",
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
                    width: "100px",
                    height: "100px",
                    background: `radial-gradient(circle, transparent 40px, ${COLORS.GREY} 40.5px)`,
                    pointerEvents: "auto",
                  }}
                />
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export const render = createRender(Board);
