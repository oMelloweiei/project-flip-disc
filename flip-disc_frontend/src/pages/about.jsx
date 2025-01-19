import React from "react";
import FadeInSection from "../components/Fadein.jsx"; // Make sure to import your FadeInSection component
import image1 from "../assets/images/about-us1.jpg";
import "../style.css";

function About() {
  return (
    <div className="about-container bg-dark relative">
      {/* พื้นหลังส่วนบน */}
      <div className="z-0 min-h-[100vh] top-0 left-0 w-full absolute"></div>

      {/* ส่วนข้อมูล */}
      <div className="z-1 flex-col bg-white shadow-md rounded-t-7xl p-6 sm:min-h-[80vh] h-auto absolute top-[70vh] left-0 w-full">
        {/* หัวข้อหลัก */}
        <p className="block mx-auto text-center mt-1 text-3xl leading-tight font-medium my-10 text-black bg-yellow-300 p-4 rounded-lg max-w-fit">
          เกี่ยวกับโปรเจ็ค Flip-Disc
        </p>

        {/* ส่วนที่ 1: เป้าหมายของโปรเจ็ค */}
        <div className="flex flex-col md:flex-row justify-between gap-x-10 my-10 items-center">
          {/* ชุดข้อความ */}
          <FadeInSection>
            <div className="uppercase min-w-[40vw] tracking-wide text-sm text-indigo-500 font-semibold md:ml-10">
              <h1 className="block mt-1 text-3xl leading-tight font-medium text-black">
                เป้าหมายของโปรเจ็ค
              </h1>
              <p className="mt-4 text-gray-700">
                โปรเจ็ค Flip-Disc
                มีเป้าหมายเพื่อแสดงให้เห็นถึงศักยภาพของเทคโนโลยี DIY
                ในการสร้างสรรค์หน้าจอแสดงผลที่ไม่เหมือนใคร
                โดยผสมผสานความคิดสร้างสรรค์เข้ากับการออกแบบอิเล็กทรอนิกส์อย่างลงตัว
              </p>
            </div>
          </FadeInSection>

          {/* ชุดรูปภาพ */}
          <FadeInSection>
            <div className="md-shrink-0 content-center min-w-[50%]">
              <img
                className="object-contain w-full h-auto rounded-lg"
                src={image1}
                alt="เป้าหมายของโปรเจ็ค"
              />
            </div>
          </FadeInSection>
        </div>

        {/* ส่วนที่ 2: ความเป็นมาของเทคโนโลยี */}
        <div className="flex flex-col md:flex-row justify-between gap-x-10 my-10 items-center">
          {/* ชุดข้อความ */}
          <FadeInSection>
            <div className="uppercase min-w-[40vw]  tracking-wide text-sm text-indigo-500 font-semibold md:ml-10">
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
          </FadeInSection>

          {/* ชุดรูปภาพ */}
          <FadeInSection>
            <div className="md-shrink-0 content-center min-w-[50%]">
              <img
                className="object-contain w-full h-auto rounded-lg"
                src={image1}
                alt="ความเป็นมาของเทคโนโลยี Flip-Disc"
              />
            </div>
          </FadeInSection>
        </div>

        {/* ส่วนที่ 3: การนำไปใช้ในชีวิตประจำวัน */}
        <div className="flex flex-col md:flex-row justify-between gap-x-10 my-10 items-center">
          {/* ชุดข้อความ */}
          <FadeInSection>
            <div className="uppercase min-w-[40vw] tracking-wide text-sm text-indigo-500 font-semibold md:ml-10">
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
          </FadeInSection>

          {/* ชุดรูปภาพ */}
          <FadeInSection>
            <div className="md-shrink-0 content-center min-w-[50%]">
              <img
                className="object-contain w-full h-auto rounded-lg"
                src={image1}
                alt="การนำไปใช้ในชีวิตประจำวัน"
              />
            </div>
          </FadeInSection>
        </div>
      </div>
    </div>
  );
}

export default About;
