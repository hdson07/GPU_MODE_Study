import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, Brush } from 'recharts';

const SquarePerformanceChart = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    const fetchData = async () => {
      try {
        // 파일 읽기
        const fileContent = await window.fs.readFile('paste.txt', { encoding: 'utf8' });
        
        // 데이터 파싱
        const rows = fileContent.split('\n');
        const parsedData = [];
        
        // 첫 번째 행은 헤더이므로 건너뛰기
        for (let i = 1; i < rows.length; i++) {
          const row = rows[i].trim();
          if (!row) continue;
          
          // 공백으로 분리하고 빈 문자열 제거
          const columns = row.split(/\s+/).filter(col => col.length > 0);
          
          if (columns.length >= 5) {
            parsedData.push({
              index: parseInt(columns[0]),
              N: parseFloat(columns[1]),
              Triton: parseFloat(columns[2]),
              Torch_native: parseFloat(columns[3]),
              Torch_compiled: parseFloat(columns[4])
            });
          }
        }
        
        setData(parsedData);
        setLoading(false);
      } catch (error) {
        console.error('데이터 로딩 중 오류:', error);
        setLoading(false);
      }
    };
    
    fetchData();
  }, []);
  
  // 특이점 (큰 변화가 있는 N 값)
  const specialPoint = 512; // Torch (compiled)의 급격한 변화가 있는 지점
  
  if (loading) {
    return <div className="flex justify-center items-center h-64">데이터 로딩 중...</div>;
  }
  
  return (
    <div className="flex flex-col space-y-6">
      <h2 className="text-xl font-bold text-center">square() 성능 비교: Triton vs Torch (native) vs Torch (compiled)</h2>
      
      <div className="w-full bg-white p-4 rounded-lg shadow">
        <ResponsiveContainer width="100%" height={500}>
          <LineChart
            data={data}
            margin={{ top: 20, right: 30, left: 40, bottom: 50 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="N" 
              label={{ value: 'N (입력 크기)', position: 'insideBottom', offset: -20 }} 
              tick={{ fontSize: 12 }}
            />
            <YAxis 
              label={{ value: '성능 측정값', angle: -90, position: 'insideLeft', offset: -20 }} 
              tick={{ fontSize: 12 }}
              domain={['auto', 'auto']}
            />
            <Tooltip 
              formatter={(value) => value.toFixed(2)} 
              labelFormatter={(value) => `N = ${value}`}
            />
            <Legend verticalAlign="top" height={36} />
            
            <ReferenceLine x={specialPoint} stroke="red" strokeDasharray="3 3" label={{ value: '특이점', position: 'top', fill: 'red' }} />
            
            <Line 
              type="monotone" 
              dataKey="Triton" 
              name="Triton"
              stroke="#8884d8" 
              dot={false} 
              strokeWidth={2}
              activeDot={{ r: 6 }}
            />
            <Line 
              type="monotone" 
              dataKey="Torch_native" 
              name="Torch (native)"
              stroke="#82ca9d" 
              dot={false} 
              strokeWidth={2}
              activeDot={{ r: 6 }}
            />
            <Line 
              type="monotone" 
              dataKey="Torch_compiled" 
              name="Torch (compiled)"
              stroke="#ff7300" 
              dot={false} 
              strokeWidth={2}
              activeDot={{ r: 6 }}
            />
            <Brush dataKey="N" height={30} stroke="#8884d8" />
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
            {data.slice(0, 20).map((row) => (
              <tr key={row.index} className={`border-t hover:bg-gray-50 ${row.N === specialPoint ? 'bg-red-100' : ''}`}>
                <td className="p-2">{row.index}</td>
                <td className="p-2 text-right">{row.N.toFixed(1)}</td>
                <td className="p-2 text-right">{row.Triton.toFixed(2)}</td>
                <td className="p-2 text-right">{row.Torch_native.toFixed(2)}</td>
                <td className="p-2 text-right">{row.Torch_compiled.toFixed(2)}</td>
              </tr>
            ))}
            <tr className="border-t">
              <td colSpan="5" className="p-2 text-center text-gray-500">
                (처음 20개 행만 표시됨 - 총 {data.length}개 행)
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      
      <div className="p-4 bg-white rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-2">데이터 분석 요약</h3>
        <ul className="list-disc pl-5 space-y-1">
          <li>
            <strong>Torch (compiled)</strong>는 <span className="font-medium text-red-600">N=512</span>에서 급격한 성능 변화가 있습니다 
            (39.11 → 232.09, 증가량: 약 192.98).
          </li>
          <li>
            <strong>Triton</strong>은 N이 커질수록 대체로 안정적인 성능을 보이며, 225-232 범위에서 유지됩니다.
          </li>
          <li>
            <strong>Torch (native)</strong>도 211-234 범위에서 안정적인 성능을 보입니다.
          </li>
          <li>
            <strong>Torch (compiled)</strong>는 초기 값이 낮다가 급격히 증가한 후 242-243 범위에서 안정화됩니다.
          </li>
        </ul>
      </div>
    </div>
  );
};

export default SquarePerformanceChart;