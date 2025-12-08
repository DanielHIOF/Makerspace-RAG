import React from 'react';

function LoadingIndicator() {
  return (
    <div className="message assistant" role="status" aria-label="Laster svar">
      <div className="typing">
        <span></span>
        <span></span>
        <span></span>
      </div>
    </div>
  );
}

export default LoadingIndicator;
