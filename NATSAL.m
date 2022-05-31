clear

opts = delimitedTextImportOptions("NumVariables", 3);

% Specify range and delimiter
opts.DataLines = [3, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["NumberinEngland", "Age", "NumberNewSamesexPartnersinlastyear"];
opts.VariableTypes = ["double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
tbl = readtable("NATSAL3_MSM.csv", opts);

% Convert to output type
N = tbl.NumberinEngland;
Age = tbl.Age;
P = tbl.NumberNewSamesexPartnersinlastyear;

clear opts tbl

%%
clear PP
k=0;
Required_Pop=1e6;  % Run on a subsample of English population.

for i=1:length(N)
    n=round(N(i)*Required_Pop/56e6);
    PP(k+[1:n])=random('Poisson',P(i)*ones(1,n));
    k=k+n;
end
PP(PP==0)=1;
keepPP=PP;
%%
Scale=1.0;

fun = @(x,y) 1e-6+exp(-Scale*(x.^0.3-y^0.3).^2);

NN=length(PP);
M=spalloc(NN,NN,2*sum(PP));

[y,O]=sort(keepPP,'descend');

PP=keepPP;
for ii=1:length(PP)
    i=O(ii);
    QQ=PP;
    QQ(i)=0;
    F=fun(keepPP,keepPP(i));
    for k=1:PP(i)
        if sum(QQ)>0
            j=Pick_From_Vector(QQ.*F,1);
            M(i,j)=1; M(j,i)=1;
            QQ(j)=0; PP(j)=PP(j)-1;
        end
    end
    PP(i)=0;
end

%% Run an outbreak

Infectivity_Profile=[0 0 0 0.2 0.2 0.2 0.3 0.5 0.8 1 1 1 1 1 1 1 1 1 0.8 0.5 0]; l=length(Infectivity_Profile)+1;
        
Prob=[0.1:0.1:0.9];

R_of_Random_Inf=[0.1];  % has to be less than 1.


for p=1:length(Prob)
    
    for loop=1:100
        
        State=zeros(length(PP),1);
        
        Trans=Prob(p)/365; % Prob chance of infection, and partners are per year.
        
        State(O(1))=1; % start first infection
        
        Nstate=zeros(1,length(Infectivity_Profile)+1);
        
        for time=1:200
            
            FoI=zeros(length(PP),1);
            FoI(State>0 & State<l)=Infectivity_Profile(State(State>0 & State<l));
            FoI=Trans*M*FoI; FoI(State>0)=0;
            
            New_random_inf=random('Poisson',(R_of_Random_Inf/sum(Infectivity_Profile))*...
                (sum(Infectivity_Profile(State(State>0 & State<l))) + ...
                sum(Nstate(1:length(Infectivity_Profile)).*Infectivity_Profile)));
            
             
            State(State>0 & State<l)=1+State(State>0 & State<l); % Move people through infection states.
            State(FoI>rand(length(PP),1))=1; % Generate infections
           
            Nstate(end)=Nstate(end)+Nstate(end-1);
            Nstate(2:(end-1))=Nstate(1:(end-2));
            Nstate(1)=New_random_inf;
            
            
            Inf_Total(time) = sum(State>0 & State<l);
            Rec_Total(time) = sum(State==l);
            Inf_New(time) = sum(State==1);
        end
        
        HS=hist(keepPP(State==0),[1:max(keepPP)]);
        HR=hist(keepPP(State==l),[1:max(keepPP)]);
        
        GR=mean(Inf_New(4:44)./Inf_Total(3:43));
        
        EpiSize(loop,p)=Rec_Total(end);
        NEpiSize(loop,p)=Nstate(end);
    end
end

%%
for p=1:length(Prob)
    CI=prctile(EpiSize(:,p)*56e6/Required_Pop,[2.5 97.5]);
    plot(Prob(p)+0.04*[-1 1 0 0 1 -1],CI([1 1 1 2 2 2]),'-k','LineWidth',2); hold on
    CI=prctile(EpiSize(:,p)*56e6/Required_Pop,[25 75]);
    plot(Prob(p)+0.03*[-1 1 1 -1 -1],CI([1 1 2 2 1]),'-k','LineWidth',2); hold on
    H=plot(Prob(p),mean(EpiSize(:,p)*56e6/Required_Pop),'.k','MarkerSize',20); hold on
    
    OffSet=0.01;
    CI=prctile(NEpiSize(:,p)*56e6/Required_Pop,[2.5 97.5]);
    plot(OffSet+Prob(p)+0.04*[-1 1 0 0 1 -1],CI([1 1 1 2 2 2]),'-r','LineWidth',2); hold on
    CI=prctile(NEpiSize(:,p)*56e6/Required_Pop,[25 75]);
    plot(OffSet+Prob(p)+0.03*[-1 1 1 -1 -1],CI([1 1 2 2 1]),'-r','LineWidth',2); hold on
    h=plot(OffSet+Prob(p),mean(NEpiSize(:,p)*56e6/Required_Pop),'.r','MarkerSize',20); hold on
end
hold off
set(gca,'FontSize',14);
legend([H h],'MSM population','General Population','FontSize',20,'Location','NorthWest');
xlabel('Risk of Transmission for sexual contact','FontSize',18);








