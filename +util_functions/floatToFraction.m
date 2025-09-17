function fractionStr = floatToFraction(x)
   % Convert a floating-point number to its fraction representation as a string
  
   % Use the rat function to approximate the fraction
   [num, den] = rat(x);
  
   % Convert to string format
   if den == 1
       fractionStr = string(num);
   else
       fractionStr = num2str(num) + "/" + num2str(den);
   end
end