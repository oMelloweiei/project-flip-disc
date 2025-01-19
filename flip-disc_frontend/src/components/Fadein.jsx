import React from "react";

function FadeInSection(props) {
  const [isVisible, setVisible] = React.useState(false); // เริ่มต้นด้วย false
  const domRef = React.useRef();

  React.useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          setVisible(entry.isIntersecting); // ถ้าเข้ามาใน viewport จะ set ให้เป็น true
        });
      },
      {
        threshold: 0.25, // ใช้ threshold เพื่อกำหนดว่าเมื่อ 50% ของ element เข้ามาใน viewport ก็ให้ถือว่าเป็น "visible"
      }
    );

    observer.observe(domRef.current);
    return () => observer.unobserve(domRef.current); // ทำความสะอาดเมื่อ component unmount
  }, []);

  return (
    <div
      className={`fade-in-section ${isVisible ? "is-visible" : ""}`}
      ref={domRef}
    >
      {props.children}
    </div>
  );
}

export default FadeInSection;
