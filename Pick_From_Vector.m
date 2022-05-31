function X=Pick_From_Vector(P,N)

C=cumsum(P); C=[0 C]/C(end);

R=rand(1,N);

for i=N:-1:1
    X(i)=find(C<=R(i),1,'last');
end
