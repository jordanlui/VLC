function position=XY(V1,V2,V3,V4)
DB=;
T1=DB(:,1);
T2=DB(:,2);
T3=DB(:,3);
T4=DB(:,4);
[m,n]=size(DB);
for i=1:m
    Vidiff=abs(V1-T1(i));
for j=i+1:m
        Vjdiff=abs(V1-T1(j));
        if (Vidiff<=Vjdiff)
            k=find(T1==T1(i));
            X1=DB(k,5);
            Y1=DB(k,6);
        end
  end
end
for i=1:m
    Vidiff=abs(V2-T2(i));
for j=i+1:m
        Vjdiff=abs(V2-T2(j));
        if (Vidiff<=Vjdiff)
            k=find(T2==T2(i));
            X2=DB(k,5);
            Y2=DB(k,6);
        end
  end
end
for i=1:m
    Vidiff=abs(V3-T3(i));
for j=i+1:m
        Vjdiff=abs(V3-T3(j));
        if (Vidiff<=Vjdiff)
            k=find(T3==T3(i));
            X3=DB(k,5);
            Y3=DB(k,6);
        end
end
end
for i=1:m
    Vidiff=abs(V4-T4(i));
for j=i+1:m
        Vjdiff=abs(V4-T4(j));
        if (Vidiff<=Vjdiff)
            k=find(T4==T4(i));
            X4=DB(k,5);
            Y4=DB(k,6);
       end
end
end
X=(X1+X2+X3+X4)/4;
Y=(Y1+Y2+Y3+Y4)/4;
position=[X;Y];
end