import React, { useEffect, useRef } from "react";
import image1 from "../assets/images/about-us1.jpg";
import "../style.css";

function About() {
  const sectionRefs = [useRef(null), useRef(null), useRef(null)];

  useEffect(() => {
    const options = {
      root: null, // ใช้ viewport
      rootMargin: "0px",
      threshold: 0.5, // เริ่มต้นทำงานเมื่อ element เข้ามา 50% ใน viewport
    };

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
        } else {
          entry.target.classList.remove("is-visible");
        }
      });
    }, options);

    sectionRefs.forEach((ref) => {
      if (ref.current) {
        observer.observe(ref.current);
      }
    });

    // Clean up observer
    return () => {
      sectionRefs.forEach((ref) => {
        if (ref.current) {
          observer.unobserve(ref.current);
        }
      });
    };
  }, []);

  return (
    <div className="about-container bg-dark">
      {/* พื้นหลังส่วนบน */}
      <div className="z-0 bg-orange-600 min-h-[70vh]"></div>

      {/* ส่วนข้อมูล */}
      <div className="z-1 flex-col bg-white shadow-md rounded-t-7xl p-6 sm:min-h-[80vh] h-auto">
        {/* หัวข้อหลัก */}
        <p className="block mx-auto text-center mt-1 text-3xl leading-tight font-medium my-10 text-black bg-yellow-300 p-4 rounded-lg max-w-fit">
          เกี่ยวกับโปรเจ็ค Flip-Disc
        </p>

        {/* ส่วนที่ 1: เป้าหมายของโปรเจ็ค */}
        <div
          ref={sectionRefs[0]}
          className="fade-in-section flex flex-col md:flex-row justify-between gap-x-10 my-10 items-center"
        >
          <div className="uppercase min-w-[40%] tracking-wide text-sm text-indigo-500 font-semibold md:ml-10">
            <h1 className="block mt-1 text-3xl leading-tight font-medium text-black">
              เป้าหมายของโปรเจ็ค
            </h1>
            <p className="mt-4 text-gray-700">
              โปรเจ็ค Flip-Disc มีเป้าหมายเพื่อแสดงให้เห็นถึงศักยภาพของเทคโนโลยี
              DIY ในการสร้างสรรค์หน้าจอแสดงผลที่ไม่เหมือนใคร
              โดยผสมผสานความคิดสร้างสรรค์เข้ากับการออกแบบอิเล็กทรอนิกส์อย่างลงตัว
            </p>
          </div>
          <div className="md-shrink-0 content-center min-w-[50%]">
            <img
              className="object-contain w-full h-auto rounded-lg"
              src={image1}
              alt="เป้าหมายของโปรเจ็ค"
            />
          </div>
        </div>

        {/* ส่วนที่ 2: ความเป็นมาของเทคโนโลยี */}
        <div
          ref={sectionRefs[1]}
          className="fade-in-section flex flex-col md:flex-row justify-between gap-x-10 my-10 items-center"
        >
          <div className="uppercase min-w-[40%] tracking-wide text-sm text-indigo-500 font-semibold md:ml-10">
            <h1 className="block mt-1 text-3xl leading-tight font-medium text-black">
              ความเป็นมาของเทคโนโลยี Flip-Disc
            </h1>
            <p className="mt-4 text-gray-700">
              เทคโนโลยี Flip-Disc เริ่มต้นจากการใช้งานในระบบป้ายจอแสดงผล เช่น
              ป้ายรถไฟและสนามบิน
              ซึ่งใช้หลักการทางแม่เหล็กไฟฟ้าเพื่อเปลี่ยนหน้าของจานสีดำและสีขาว
              เพื่อแสดงข้อมูลได้ชัดเจนแม้ในที่แสงจ้า
            </p>
          </div>
          <div className="md-shrink-0 content-center min-w-[50%]">
            <img
              className="object-contain w-full h-auto rounded-lg"
              src={image1}
              alt="ความเป็นมาของเทคโนโลยี Flip-Disc"
            />
          </div>
        </div>

        {/* ส่วนที่ 3: การนำไปใช้ในชีวิตประจำวัน */}
        <div
          ref={sectionRefs[2]}
          className="fade-in-section flex flex-col md:flex-row justify-between gap-x-10 my-10 items-center"
        >
          <div className="uppercase min-w-[40%] tracking-wide text-sm text-indigo-500 font-semibold md:ml-10">
            <h1 className="block mt-1 text-3xl leading-tight font-medium text-black">
              การนำไปใช้ในชีวิตประจำวัน
            </h1>
            <p className="mt-4 text-gray-700">
              ปัจจุบันเทคโนโลยี Flip-Disc ไม่เพียงใช้ในระบบป้ายเท่านั้น
              แต่ยังถูกพัฒนาสำหรับงานศิลปะเชิงโต้ตอบ
              การจัดแสดงสินค้าในงานนิทรรศการ
              และสร้างประสบการณ์ที่น่าจดจำให้กับผู้ชมในทุกมิติ
            </p>
          </div>
          <div className="md-shrink-0 content-center min-w-[50%]">
            <img
              className="object-contain w-full h-auto rounded-lg"
              src={image1}
              alt="การนำไปใช้ในชีวิตประจำวัน"
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default About;
