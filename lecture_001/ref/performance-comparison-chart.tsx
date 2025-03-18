import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Label } from 'recharts';
import Papa from 'papaparse';

const PerformanceChart = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    const fetchData = async () => {
      try {
        // 파일 내용 읽기
        const response = await window.fs.readFile('paste.txt', { encoding: 'utf8' });
        
        // 데이터 라인만 추출 (헤더 제외)
        const dataLines = response.split('\n').slice(3);
        const csvData = dataLines.join('\n');
        
        // 데이터 파싱
        const parsedData = Papa.parse(csvData, {
          header: false,
          skipEmptyLines: true,
          delimiter: ' ',
          transform: function(value) {
            return value.trim();
          }
        });
        
        // 데이터 정리
        const cleanedData = parsedData.data.map(row => {
          // 빈 문자열 필터링
          const filtered = row.filter(cell => cell !== '');
          
          // 데이터가 있으면 반환
          if (filtered.length >= 5) {
            return {
              index: parseInt(filtered[0]),
              N: parseFloat(filtered[1]),
              Triton: parseFloat(filtered[2]),
              'Torch_native': parseFloat(filtered[3]),
              'Torch_compiled': parseFloat(filtered[4])
            };
          }
          return null;
        }).filter(item => item !== null);
        
        setData(cleanedData);
        setLoading(false);
      } catch (error) {
        console.error('Error loading data:', error);
        setLoading(false);
      }
    };
    
    fetchData();
  }, []);
  
  if (loading) {
    return <div className="flex justify-center items-center h-64">데이터 로딩 중...</div>;
  }
  
  return (
    <div className="flex flex-col space-y-6">
      <h2 className="text-xl font-bold text-center">성능 비교: Triton vs Torch (native) vs Torch (compiled)</h2>
      
      <div className="w-full bg-white p-4 rounded-lg shadow">
        <ResponsiveContainer width="100%" height={500}>
          <LineChart
            data={data}
            margin={{ top: 20, right: 30, left: 30, bottom: 30 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="N" 
              label={{ value: 'N (입력 크기)', position: 'insideBottom', offset: -10 }} 
              tick={{ fontSize: 12 }}
            />
            <YAxis 
              label={{ value: '성능 측정값', angle: -90, position: 'insideLeft' }}
              tick={{ fontSize: 12 }}
            />
            <Tooltip formatter={(value) => value.toFixed(2)} />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="Triton" 
              stroke="#8884d8" 
              dot={false} 
              strokeWidth={2}
              activeDot={{ r: 6 }}
            />
            <Line 
              type="monotone" 
              dataKey="Torch_native" 
              stroke="#82ca9d" 
              dot={false} 
              strokeWidth={2}
              activeDot={{ r: 6 }}
            />
            <Line 
              type="monotone" 
              dataKey="Torch_compiled" 
              stroke="#ff7300" 
              dot={false} 
              strokeWidth={2}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      
      <div className="overflow-x-auto">
        <table className="min-w-full bg-white shadow-md rounded-lg overflow-hidden">
          <thead className="bg-gray-100">
            <tr>
              <th className="p-2 text-left">행 번호</th>
              <th className="p-2 text-right">N</th>
              <th className="p-2 text-right">Triton</th>
              <th className="p-2 text-right">Torch (native)</th>
              <th className="p-2 text-right">Torch (compiled)</th>
            </tr>
          </thead>
          <tbody>
            {data.map((row) => (
              <tr key={row.index} className="border-t hover:bg-gray-50">
                <td className="p-2">{row.index}</td>
                <td className="p-2 text-right">{row.N.toFixed(1)}</td>
                <td className="p-2 text-right">{row.Triton.toFixed(2)}</td>
                <td className="p-2 text-right">{row.Torch_native.toFixed(2)}</td>
                <td className="p-2 text-right">{row.Torch_compiled.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default PerformanceChart;